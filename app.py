from fastapi import FastAPI, HTTPException, Query
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json
import re
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import uuid

load_dotenv()

app = FastAPI()

# Email sender function
def send_email(recipient_email: str, subject: str, body: str, pdf_filepath: str = None):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_password = os.getenv("EMAIL_PASSWORD")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach PDF if provided
    if pdf_filepath:
        try:
            with open(pdf_filepath, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{os.path.basename(pdf_filepath)}"'
                )
                msg.attach(part)
        except Exception as e:
            print(f"Error attaching PDF: {e}")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}")

# PDF generator function - modified to include query results details
def generate_invoice_pdf(data: list[dict], query: str) -> str:
    filename = f"invoice_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join("invoices", filename)
    os.makedirs("invoices", exist_ok=True)

    c = canvas.Canvas(filepath, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Invoice Report")
    c.drawString(50, 735, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, 720, f"Query: {query}")

    y = 690
    if data:
        # Assuming that the data is a list of dictionaries representing items like name, price, etc.
        headers = list(data[0].keys())
        c.drawString(50, y, " | ".join(headers))
        y -= 20
        for row in data:
            row_str = " | ".join(str(row.get(col, '')) for col in headers)
            c.drawString(50, y, row_str)
            y -= 20
            if y < 100:
                c.showPage()
                y = 750
    else:
        c.drawString(50, y, "No data available.")

    c.save()
    return filepath

@app.get("/")
def read_root():
    return {"hello": "world"}

def configure_db(host: str, user: str, password: str, database: str):
    try:
        conn_string = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_string)
        return SQLDatabase(engine), engine
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def extract_sql_query(agent_response):
    if re.match(r'^[\d.]+$', agent_response.strip()):
        raise ValueError(f"Agent returned a numeric value instead of SQL: {agent_response}")
    sql_match = re.search(r'```sql\s*(.*?)\s*```', agent_response, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    sql_match = re.search(r'`(.*?)`', agent_response, re.DOTALL)
    if sql_match:
        return sql_match.group(1).strip()
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'SHOW']
    for keyword in sql_keywords:
        sql_match = re.search(f'{keyword}\\s+.*', agent_response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            return sql_match.group(0).strip()
    if any(keyword in agent_response.upper() for keyword in sql_keywords):
        return agent_response.strip()
    raise ValueError(f"Could not identify SQL query in agent response: {agent_response}")

def is_valid_sql(query):
    if re.match(r'^[\d.]+$', query.strip()):
        return False
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'JOIN', 'GROUP BY', 'ORDER BY']
    return any(keyword.upper() in query.upper() for keyword in sql_keywords)

@app.post("/chat")
async def chat_with_db(
    host: str = Query(...),
    user: str = Query(...),
    password: str = Query(...),
    database: str = Query(...),
    query: str = Query(...)
):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            streaming=False
        )

        db, engine = configure_db(host, user, password, database)

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )

        sql_generation_prompt = f"""
        Generate a valid SQL query to answer this question.
        Question: "{query}"
        Return ONLY the SQL query without explanation or formatting.
        """

        agent_response = agent.run(sql_generation_prompt)
        try:
            sql_query = extract_sql_query(agent_response)
            if not is_valid_sql(sql_query):
                raise ValueError("Invalid SQL")
        except Exception:
            retry_prompt = f"""Retry. Generate a valid SQL query for this question: "{query}"."""
            agent_response = agent.run(retry_prompt)
            sql_query = extract_sql_query(agent_response)

        sql_result_list = []
        with engine.connect() as connection:
            result = connection.execute(text(sql_query))
            if result.returns_rows:
                columns = result.keys()
                for row in result:
                    row_dict = {col: (str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value)
                                for col, value in zip(columns, row)}
                    sql_result_list.append(row_dict)
                sql_result_str = json.dumps(sql_result_list)
            else:
                sql_result_str = "Query executed successfully. No rows returned."

        sent_to = set()
        if any(word in query.lower() for word in ["pending", "overdue", "remind", "email"]):
            for row in sql_result_list:
                email = row.get("email")
                if email and email not in sent_to:
                    # Attach PDF when sending email
                    pdf_path = None
                    if "pdf" in query.lower() or "invoice" in query.lower():
                        pdf_path = generate_invoice_pdf(sql_result_list, query)

                    send_email(
                        recipient_email=email,
                        subject="Reminder: Pending Invoice",
                        body="Dear Customer,\n\nThis is a reminder that you have a pending invoice. Please clear it soon.\n\nThanks.",
                        pdf_filepath=pdf_path  # Attach the generated PDF here
                    )
                    sent_to.add(email)

        # 🧾 PDF invoice generation logic
        pdf_path = None
        if "pdf" in query.lower() or "invoice" in query.lower():
            pdf_path = generate_invoice_pdf(sql_result_list, query)

        summary_prompt = f"""
        Question: {query}
        SQL Query: {sql_query}
        SQL Result: {sql_result_str}

        Summarize the results in plain language.
        """
        summary = llm.invoke(summary_prompt).content

        return {
            "user_query": query,
            "sql_query": sql_query,
            "sql_result": sql_result_list if result.returns_rows else "Query executed successfully. No rows returned.",
            "summary": summary,
            "pdf_invoice_path": pdf_path if pdf_path else "No PDF generated"
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)