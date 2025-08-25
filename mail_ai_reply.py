

# First, run these commands to install the required libraries.
# !pip install gradio transformers torch google-api-python-client google-auth-oauthlib google-auth-httplib2 --q
# Make sure to restart the runtime after installation if needed.

# --- Imports and Authentication ---
import os
import pickle
import base64
from email.mime.text import MIMEText

from google.colab import drive
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import gradio as gr

# Mount Google Drive to access credentials and token files
try:
    drive.mount('/content/drive')
except Exception as e:
    print(f"Failed to mount Google Drive: {e}. Please ensure you are running in a Colab environment.")

# --- The model configuration remains the same ---
model_name = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
service = None # Initialize service globally

def gmail_authenticate():
    """Authenticates with Gmail and returns the service object."""
    global service
    creds = None
    token_path = "/content/drive/MyDrive/gmail_bot/config/token.pkl"
    credentials_path = "/content/drive/MyDrive/gmail_bot/config/credentials.json"

    print("Attempting to authenticate Gmail API...")

    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds)
    print("Gmail API authenticated successfully.")
    return "Gmail API authenticated successfully."

# --- Functions to fetch emails from different labels ---
def fetch_emails(label_ids, query=None):
    """Fetches emails from a specified label with an optional query."""
    try:
        if service is None:
            return "Authentication is required. Please check your credentials.", [], []

        results = service.users().messages().list(
            userId="me",
            labelIds=label_ids,
            q=query,
            maxResults=10
        ).execute()

        messages = results.get("messages", [])

        if not messages:
            return f"📭 No emails found.", [], []

        email_list = []
        for msg in messages:
            email_info = service.users().messages().get(userId="me", id=msg["id"]).execute()
            headers = email_info["payload"]["headers"]

            sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
            snippet = email_info.get("snippet", "")

            email_list.append({"id": msg["id"], "from": sender, "subject": subject, "snippet": snippet})

        email_data_for_gradio = [[email['subject'], email['from'], email['snippet']] for email in email_list]
        email_ids = [email['id'] for email in email_list]

        return f"Emails fetched successfully!", email_data_for_gradio, email_ids

    except Exception as e:
        print(f"Error fetching emails: {e}")
        return f"❌ An error occurred: {str(e)}", [], []

def get_email_body_and_from(email_id, email_from):
    """Retrieves the full body of a selected email and returns it along with the sender."""
    try:
        if not email_id or service is None:
            return "Please select an email first.", ""

        msg = service.users().messages().get(userId='me', id=email_id, format='full').execute()

        payload = msg['payload']
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body']['data']
                    text = base64.urlsafe_b64decode(data).decode('utf-8')
                    return text, email_from
        elif 'body' in payload and 'data' in payload['body']:
            data = payload['body']['data']
            text = base64.urlsafe_b64decode(data).decode('utf-8')
            return text, email_from

        return "Could not retrieve email body.", email_from

    except Exception as e:
        print(f"Error getting email body: {e}")
        return f"❌ An error occurred while retrieving email body: {str(e)}", email_from

def generate_ai_reply(email_text):
    """Generates an AI reply based on the email text."""
    if not email_text:
        return "Please select an email to generate a reply."

    print("🤖 Generating AI reply...")
    prompt = f"Based on the following email, generate a polite and concise email reply. The output should only be the body of the email, without a subject line or a new greeting.\n\nEmail: {email_text}\n\nReply:"

    # Reduced max_new_tokens for a faster, more concise reply
    response = generator(prompt, max_new_tokens=30, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    generated_text = response[0]["generated_text"]
    reply_start_index = generated_text.find("Reply:")
    if reply_start_index != -1:
        return generated_text[reply_start_index + len("Reply:"):].strip()
    return generated_text.strip()

def send_reply(email_id, to_email, subject, body):
    """Sends the reply and marks the email as read."""
    try:
        if service is None:
            return "Authentication is required before sending."

        message = MIMEText(body)
        message['to'] = to_email
        message['subject'] = "Re: " + subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        service.users().messages().send(userId='me', body={'raw': raw_message}).execute()

        service.users().messages().modify(
            userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()

        return "✅ Reply sent and email marked as read."

    except Exception as e:
        print(f"Error sending reply: {e}")
        return f"❌ An error occurred: {str(e)}"

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# <center>AI Email Assistant</center>")
    gr.Markdown("---")

    # State variables to hold data
    email_ids_state = gr.State([])
    email_data_state = gr.State([])

    with gr.Row():
        # Left-hand column for navigation
        with gr.Column(scale=1):
            gr.Markdown("### Mailboxes")
            inbox_btn = gr.Button("📬 Inbox")
            inbox_unread_btn = gr.Button("🆕 Unread Mail")
            sent_btn = gr.Button("📧 Sent Mail")
            starred_btn = gr.Button("⭐ Starred")

            status_message = gr.Textbox(label="Status", interactive=False)

        # Right-hand column for content
        with gr.Column(scale=4):
            gr.Markdown("### Email List")
            email_df = gr.Dataframe(
                headers=["Subject", "From", "Snippet"],
                type="array",
                interactive=False,
                label="Emails"
            )
            email_dropdown = gr.Dropdown(label="Select an Email to View")

            gr.Markdown("---")

            gr.Markdown("### Email Content & Reply")
            full_email_body = gr.Textbox(label="Full Email Body", lines=10, interactive=False)

            selected_from_state = gr.State("")
            selected_subject_state = gr.State("")
            selected_id_state = gr.State("")

            with gr.Row():
                generate_reply_btn = gr.Button("Generate AI Reply")

            ai_reply_textbox = gr.Textbox(label="AI Generated Reply", lines=5)
            gr.Markdown("You can modify the generated text above before sending.")

            with gr.Row():
                send_reply_btn = gr.Button("Send Reply")

            send_status = gr.Textbox(label="Send Status", interactive=False)

    # --- Event Handlers ---

    def load_emails_for_label(label_name):
        """Helper function to load and format emails for a given label or query."""
        query = "is:unread" if label_name == "UNREAD" else None
        label_ids = ["INBOX"] if label_name in ["INBOX", "UNREAD"] else [label_name]

        status, data, ids = fetch_emails(label_ids=label_ids, query=query)
        choices = [f"{row[0]} (From: {row[1]})" for row in data]

        return (
            status,
            gr.update(value=data),
            gr.update(choices=choices),
            ids,
            data,
            "", "", "", ""
        )

    # Combined function for initial loading
    def initial_load():
        auth_status = gmail_authenticate()
        email_status, email_data_df, email_dropdown_choices, email_ids_state_val, email_data_state_val, *rest = load_emails_for_label("INBOX")

        combined_status = f"{auth_status}\n{email_status}"

        return (
            combined_status,
            email_data_df,
            email_dropdown_choices,
            email_ids_state_val,
            email_data_state_val,
            *rest
        )

    demo.load(
        fn=initial_load,
        inputs=[],
        outputs=[status_message, email_df, email_dropdown, email_ids_state, email_data_state,
                 full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    # Handlers for the left-side buttons
    inbox_btn.click(
        fn=lambda: load_emails_for_label("INBOX"),
        inputs=[],
        outputs=[status_message, email_df, email_dropdown, email_ids_state, email_data_state,
                 full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    inbox_unread_btn.click(
        fn=lambda: load_emails_for_label("UNREAD"),
        inputs=[],
        outputs=[status_message, email_df, email_dropdown, email_ids_state, email_data_state,
                 full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    sent_btn.click(
        fn=lambda: load_emails_for_label("SENT"),
        inputs=[],
        outputs=[status_message, email_df, email_dropdown, email_ids_state, email_data_state,
                 full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    starred_btn.click(
        fn=lambda: load_emails_for_label("STARRED"),
        inputs=[],
        outputs=[status_message, email_df, email_dropdown, email_ids_state, email_data_state,
                 full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    # Dropdown selection change
    def handle_dropdown_change(selected_text, all_emails, all_ids):
        if not selected_text:
            return "","","",""

        subject_end = selected_text.rfind(' (From:')
        subject = selected_text[:subject_end] if subject_end != -1 else selected_text

        for i, row in enumerate(all_emails):
            if row[0] == subject:
                email_id = all_ids[i]
                email_from = row[1]
                email_body, _ = get_email_body_and_from(email_id, email_from)
                return email_body, email_id, email_from, subject

        return "Email not found.", "", "", ""

    email_dropdown.change(
        fn=handle_dropdown_change,
        inputs=[email_dropdown, email_data_state, email_ids_state],
        outputs=[full_email_body, selected_id_state, selected_from_state, selected_subject_state]
    )

    generate_reply_btn.click(
        fn=generate_ai_reply,
        inputs=full_email_body,
        outputs=ai_reply_textbox
    )

    send_reply_btn.click(
        fn=send_reply,
        inputs=[
            selected_id_state,
            selected_from_state,
            selected_subject_state,
            ai_reply_textbox
        ],
        outputs=send_status
    )

demo.launch(debug=True)