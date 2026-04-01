from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI(model="gpt-4o-mini")

#draft email template
draft_email_prompt_template=PromptTemplate(
    template=""""
    you are a helpful assistant. Your task is to write a draft email for the  following email.
    Topic:{email_topic} 
    Recipient:{recipient}
    name:{name}

    Give me the draft email in the following json format:
    {{
        "draft_email": "This is a draft email"
    }}
    """
) 
draft_email_chain=draft_email_prompt_template| model|JsonOutputParser()

#grammer check and humanizer template
grammar_chain_prompt_template=PromptTemplate(
    template=""""
    you are a helpful assistant. Your task is to write a draft email for the  following email.
    Email:{draft_email}
    Also humanize the email to make more natural and readable.
    Give me the grammer in the following json format:
    {{
        "grammer_check_comments": "This is the grammer check",
        "final_email": "This is a Humanized email"
    }}
    """
)

# #chain define
grammer_chain=grammar_chain_prompt_template| model|JsonOutputParser()

# # combined_chain=(draft_email_chain|grammer_chain)
# combined_chain = (
#     draft_email_chain 
#     | (lambda x: {"draft_email": x["draft_email"]}) 
#     | grammer_chain 
# )


# response=combined_chain.invoke({
#     "email_topic":"new prompt launch",
#     "recipient":"John",
#     "name":"Anil"
# })

# print(response)

combined_chain = (draft_email_chain|grammer_chain)

subject_line_prompt_template=PromptTemplate(
    template=""""
    you are a helpful assistant. Your task is to write a subject line for the  following email.
    question:{question}

    Give me the subject line in the following json format:
    {{
        "subject_line": "This is the subject line"
    }}
    """
)

subject_line_chain=subject_line_prompt_template| model|JsonOutputParser()

parallel_chain = RunnableParallel[dict](
    {
        "combined_chain": combined_chain,
        "subject_line":subject_line_chain,
    }
)

# response = parallel_chain.invoke({
#     "email_topic": "new prompt launch",
#     "recipient": "John",
#     "name": "Anil",
#     "question":"What is the capital of france?",
# })
# print(response["subject_line"])
# print(response["combined_chain"])