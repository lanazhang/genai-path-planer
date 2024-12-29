import streamlit as st
from streamlit_cognito_auth import CognitoAuthenticator
import os
from PyPDF2 import PdfReader
import json
import boto3

pool_id = st.secrets["COGNITIO_POOL_ID"] 
app_client_id = st.secrets["COGNITIO_APP_CLIENT_ID"]
aws_key = st.secrets["AWS_KEY"]
aws_secret = st.secrets["AWS_SECRET"]
aws_region = st.secrets["AWS_REGION"]

bedrock_runtime = boto3.client('bedrock-runtime',aws_access_key_id=aws_key,aws_secret_access_key=aws_secret,region_name=aws_region)

if pool_id and app_client_id:
    app_client_secret = os.environ.get("COGNITIO_APP_CLIENT_SECRET", None)


    authenticator = CognitoAuthenticator(
        pool_id=pool_id,
        app_client_id=app_client_id,
        app_client_secret=app_client_secret,
    )
    #authenticator.logout()
    is_logged_in = authenticator.login()
    st.session_state['is_logged_in'] = is_logged_in
    if not is_logged_in:
        st.stop()
    
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def call_bedrock(bedrock_client,
                          model_id,
                          system_prompts,
                          messages):
    # Inference parameters to use.
    temperature = 0.5
    top_k = 200

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_k": top_k}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    return response


if __name__ == "__main__":
    st.title("Path Planer")
    st.write("Welcome to PathPlanner (or your chosen name), your personal academic advisor for achieving your goals! Follow these simple steps to get started:")
    goal = st.pills(
        "Your goal",
        ("Get a high school diploma", "Go to college"),
    )

    grade_level = st.pills(
        "Your current high school year",
        ("Grade 9", "Grade 10", "Grade 11", "Grade 12"),
    )

    score_sat = st.number_input("Your recent SAT score", placeholder=1200, step=100, min_value=400, max_value=1500)
    st.write("or")
    score_gpa = st.selectbox("Your recent GPA score", [0,1,2,3,4,5])

    uploaded_file = st.file_uploader("Upload the school course catalog document", type="pdf")
    pdf_text = None

    if uploaded_file is not None:
        try:
            # Extract text
            pdf_text = extract_text_from_pdf(uploaded_file)
            
            # Display extracted text
            st.text_area("Text from PDF:", pdf_text, height=100)
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

        if st.button("Get recommendation"):
            # Send to LLM
            system_prompts = [{"text": """
                                You are a school advisor assisting high school students in creating a plan to achieve their goals. 
                                The student will share their objective—either graduating with a diploma or pursuing college admission, along with their current grade level, most recent standardized test scores or GPA, and a course catalog document provided by their high school. Based on this information, provide three tailored options categorized as:
                                Safety: A plan that aligns with easily attainable requirements.
                                Target: A plan that is realistic but requires consistent effort to achieve.
                                Reach: An ambitious plan requiring significant improvement or exceptional performance.
                                Include the required GPA and any additional recommendations for each option.
                                """}]
            contents = [{"text":f"My goal is {goal}"}, {"text":f"My current grade level is {grade_level}"}]
            if score_sat:
                contents.append({"text":f"My recent SAT is {score_sat}"})
            if score_gpa:
                contents.append({"text":f"My recent GPA is {score_gpa}"})
            contents.append({"text":f"The school course catalog: {pdf_text}"})

            messages=[
                        {"role": "user", "content": contents},
                    ]
            # call bedrock
            with st.spinner('Evaluating...'):
                response = call_bedrock(bedrock_runtime, "anthropic.claude-3-5-sonnet-20240620-v1:0", system_prompts, messages)
                st.write(response.get("output",{}).get("message",{}).get("content",[{}])[0].get("text"))


