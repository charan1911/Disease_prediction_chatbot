import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from langchain.chains import LLMChain
from langchain import PromptTemplate,  LLMChain
from langchain.llms import Together
from langchain.memory import ConversationBufferMemory
import os



os.environ["TOGETHER_API_KEY"] =  "8cb4ed7567a6f51c25e1287206892d78031a0185bef9195abe3c554e6037e3bb" 

system_prompt = """
You're a helpful polite medical chat assistant that helps user in knowing the disease with the symptoms provided

you have to respond just as assistant you are strictly prohibited to respond as user you just have to respond to user_input

The medical chat assistant is defined by 4 parameters:

1.Name of the user
 -Ask the name of the user

2. Symptoms
  -ask user about what kind of symptoms they are facing ask details about the symptoms ask details about symptoms don't ask everything at a time ask one question at a time.

3. Disease Identification
  - by the symptoms provided by the user Identify the disease,here provide the caution to the user that you're just a chatbot and if symptoms are severe they need to consult a doctor.

4. Precautions and measures
  - Provide the precautions and measures that can be taken,here provide the caution to the user that you're just a chatbot and if symptoms are severe they need to consult a doctor.

5. doctor
    - provide user which specialist they have to consult for the disease identified.

ask each parameter one by one don't ask everything at a time first ask the name of the user and the  ask the symptoms then ask details.

please ask questions only one at a time don't ask multiple questions in one go.

STRICLTY keep the responses short only one line STRICLTY don't generate response more than one line.

Don't pretend to be user or try to answer questions as user you just respond as assistant but not as human,you're strictly prohibited to answer as user.

User can provide the symptoms one-by-one or all at a time be intelligent enough to identify the disease properly,Ask user all the details about symptoms before identifying the disease,ask food habits,diet habits,exercising habits etc.,

you should provide a summary in the end with all the 4 parameters.
  name of the user - output name here
  Symptoms - output symptoms here
  Disease Identification - output disease identified here
  Precautions and measures - output Precautions and measures here
  doctor - output doctor here
please provide the summary with all the four parameters mentioned above."""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

instruction = "Chat History:\n\n{chat_history} \n\nHuman: {user_input}\n\n Assistant:"


template = get_prompt(instruction, system_prompt)
print(template)

def setup_chain():

    st.session_state["past"] = ['Hola!']
    st.session_state["generated"] = [
        "Hola! I'm medical Bot."
    ]

    llm = Together(model= "togethercomputer/llama-2-70b-chat",temperature=0.001,max_tokens=512)


    # Prompt 
    prompt = PromptTemplate(
    input_variables=["chat_history", "user_input"], template=template
)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory
    )

    st.session_state['chain'] = conversation
    st.session_state['memory'] = memory
    memory.clear()


def reset_state():
    st.session_state['input'] = ''
    st.session_state["past"] = ['Hola!']
    st.session_state["generated"] = [
        "Hola! I'm medical Bot."
    ]
    if('memory' in st.session_state):
        st.session_state['memory'].clear()

if not st.session_state:
    setup_chain()
    reset_state()

## generated stores AI generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = [
        "Hola! I'm medical Bot."
    ]

## past stores User's questions
if "past" not in st.session_state:
    st.session_state["past"] = ['Hola!']
    
# Layout of input/response containers
colored_header(label="", description="", color_name="blue-30")
response_container = st.container()
colored_header(label="", description="", color_name="blue-30")
input_container = st.container()

if st.button('Clear History'):
    reset_state()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("Type Below: ", "", key="input")
    return input_text


## Applying the user input box
with input_container:
    user_input = get_text()


# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(user_input):
    chain = st.session_state['chain']
    response = chain.predict(user_input = user_input)
    print(response)
    # logger.info(f'chat_history:{chat_history}')
    # db_logs.info(f"{json.dumps(message)}")
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if(user_input):
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))