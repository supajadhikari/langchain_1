from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini")

# 1. DRAFT EMAIL CHAIN 
draft_email_prompt_template = PromptTemplate(
    template="""
    You are a helpful assistant. Your task is to write a draft email.
    Topic: {email_topic} 
    Recipient: {recipient}
    Name: {name}

    Give me the draft email in the following JSON format:
    {{
        "draft_email": "This is a draft email"
    }}
    """,
    input_variables=["email_topic", "recipient", "name"]
)

# This chain produces a dictionary: {"draft_email": "..."}
draft_email_chain = draft_email_prompt_template | model | JsonOutputParser()


# 2. GRAMMAR & HUMANIZER CHAIN 
grammar_chain_prompt_template = PromptTemplate(
    template="""
    You are a helpful assistant. Your task is to polish a draft email.
    Email: {draft_email}
    
    Humanize the email to make it more natural and readable.
    Give me the response in the following JSON format:
    {{
        "grammer_check_comments": "This is the grammar check",
        "final_email": "This is a Humanized email"
    }}
    """,
    input_variables=["draft_email"]
)

# This chain expects a dictionary with the key "draft_email"
grammar_chain = grammar_chain_prompt_template | model | JsonOutputParser()


#  3. COMBINED SEQUENTIAL CHAIN 
# We use a lambda to ensure only the necessary output from draft_email_chain 
# is passed into the grammar_chain.
combined_chain = (
    draft_email_chain 
    | (lambda x: {"draft_email": x["draft_email"]}) 
    | grammar_chain
)


# --- 4. SUBJECT LINE CHAIN ---
subject_line_prompt_template = PromptTemplate(
    template="""
    You are a helpful assistant. Your task is to write a subject line for the following topic.
    Topic: {email_topic}

    Give me the subject line in the following JSON format:
    {{
        "subject_line": "This is the subject line"
    }}
    """,
    input_variables=["email_topic"]
)

subject_line_chain = subject_line_prompt_template | model | JsonOutputParser()


# 5. PARALLEL EXECUTION 
# RunnableParallel allows us to run the 'Subject Line' and the 'Email Body' logic at the same time.
# The keys "combined_chain" and "subject_line" will be the keys in our final output.
parallel_chain = RunnableParallel(
    {
        "combined_chain": combined_chain,
        "subject_line": subject_line_chain,
    }
)

# 6. EXECUTION 
if __name__ == "__main__":
    # Define inputs for all chains involved
    input_data = {
        "email_topic": "new prompt launch",
        "recipient": "John",
        "name": "SPJ"
    }

    response = parallel_chain.invoke(input_data)

    # Printing specific parts of the nested JSON response
    print(" SUBJECT LINE ")
    print(response["subject_result"]["subject_line"])
    
    print("\n HUMANIZED EMAIL ")
    print(response["body_result"]["final_email"])
    
    print("\n GRAMMAR COMMENTS ")
    print(response["body_result"]["grammer_check_comments"])