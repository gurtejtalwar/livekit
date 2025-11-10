import os
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.embeddings.create(
    input="""
 Why do core application need to be penetration tested?
 ADP wants to make sure we are offering secure core applications to our clients and prevent malicious users from unauthorized access to our
 clients’ data and credentials.
 What are the criteria to go ahead from the security standpoint?
 All critical/high issues identified in your application must be remediated. All security documentation must be submitted. No conditional
 approvals will be given.
 What information must be provided as part of security review and how?
 You must provide your organization's security policy documents (refer to Annexure A above) and details for penetration test (refer to
 Annexure C above).
 How should I share information with ADP GSO?
 You must upload the information to the shared ADP folder. GSO will provide access to this folder after your initial kickoff call.
 What are the timelines for the ADP to perform penetration tests?
 Three business weeks to perform security documentation review, initial penetration test and source code reviews
 3-4 business days for remediation test
 Timelines are subject to change if (a) testing analyst goes on emergency leave or (b) accessibility/functionality issues arise with your
 application.
 Which environment of my application should be shared for penetration tests?
 Malicious content will be used as part of the penetration tests, which might affect the data, so we recommend sharing
 staging/test/development environments for penetration tests.
 Can I make changes to the environment while a test is under way?
 You must leave the environment untouched until the penetration test is complete. No issues will be notified during midway of penetration
 test.
 What will ADP GSO provide to the partner after the initial full penetration test is completed?
 ADP GSO will share full penetration test report, which includes the issues identified, risk rating, exploitation steps of issues and remediation
 recommendations/suggestions.
 What are the timelines to remediate the critical and high issues identified?
 Generally, we recommend remediating critical issues within 7 days and high issues within 15 days.
 What are next steps after I remediate the identified issues?
 You should notify GSO once the identified critical and high issues are remediated. GSO then performs a remediation test to review whether the
 implemented remediation controls are appropriate. As part of the remediation test, only the identified critical and high issues will be reviewed.
 How many rounds of penetration tests will ADP GSO perform for each partner?
 ADP provides penetration test services free of cost. ADP GSO will perform three rounds (one initial full test and two remediation tests) for
 each partner. If there are open critical/high issues after three rounds of testing, partners must engage with a third-party vendor and submit the
 remediation test report confirming closure of issues.
 What are the next steps after the successful security review?
 Once all the critical/high issues (if any) are remediated, GSO signs off and the integration team will reach out to you with next steps.
 What is the security review process for additional enhancements after a successful initial integration?
 Refer to the "Incremental integration review requirements" section above
    """,
    model="text-embedding-3-small"
)

print(response.data[0].embedding)
# from pathlib import Path
# from dotenv import load_dotenv
# import os
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from pinecone import Pinecone

# load_dotenv()

# ###### PINECONE SETUP ######
# # Initialize Pinecone client
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
# index = pc.Index("ai-tutor")

# # Wrap in a LlamaIndex vector store
# vector_store = PineconeVectorStore(pinecone_index=index)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# # index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# print(index.describe_index_stats())