import uuid
import logging
from typing import List

from langchain_core.messages import  HumanMessage, AIMessage
from langchain_core.documents import Document
import mlflow
from mlflow.entities import SpanType


logger = logging.getLogger(__name__)


def format_docs(docs):
    return "\n\n".join(
        f"""page_title: {doc.metadata['title']} \n\
source_url: {doc.metadata['source']} \n\
date: {doc.metadata['published']} \n\
page_content: {doc.page_content}
"""
        for doc in docs
    )

class RAGChain:
    def __init__(
        self,
        llm_engine,
        prompt,
        retriever,
        experiment_name=None,
    ):
        if experiment_name:
            self.trace_collection = experiment_name
        else:
            self.trace_collection = uuid.uuid4().hex
        mlflow.set_experiment(self.trace_collection)

        prompt_input_vars = ["context", "input"]
        assert set(prompt_input_vars) == set(
            prompt.input_variables
        ), f"Prompt template must include: {prompt_input_vars}"
        
        self.llm = llm_engine
        self.prompt_template = prompt
        self.retriever = retriever


    def _retrieve(self, query: str) -> List[Document]:
        with mlflow.start_span(
            name="simple_retrieve", span_type=SpanType.RETRIEVER
        ) as span:
            span.set_inputs({"query": query})
            documents = self.retriever.invoke(query)
            span.set_outputs({"documents": documents})

            return documents


    def _generate(
        self,
        user_message: HumanMessage,
        documents: List[Document],
    ) -> AIMessage:
        formatted_messages = self.prompt_template.format_messages(
            input=user_message,
            context=format_docs(documents),
        )
        with mlflow.start_span(
            name="llm_generation", span_type=SpanType.CHAT_MODEL
        ) as span:
            span.set_inputs({"messages": formatted_messages})
            ai_message = self.llm.invoke(formatted_messages)
            span.set_outputs({"ai_message": ai_message})

        return ai_message

    def invoke(
        self, input_message: HumanMessage
    ) -> AIMessage:
        trace_name = "invoke"
        with mlflow.start_span(name=trace_name, span_type=SpanType.CHAIN) as _span:
            _span.set_inputs({"input_message": input_message})
            ai_message = self._generate(
                user_message=input_message,
                documents=self._retrieve(input_message.content),
            )

            _span.set_outputs(ai_message)

            return ai_message
