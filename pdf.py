import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.service_context import ServiceContext
from llama_index.readers.file import PDFReader

class PDFIndexer:
    def __init__(self, index_name, llm, embedded_model):
        self.index_name = index_name
        self.llm = llm
        self.embedded_model = embedded_model

    def get_index(self, data):
        index = None
        if not os.path.exists(self.index_name):
            print("building index", self.index_name)
            
            index = VectorStoreIndex.from_documents(
                data, 
                show_progress=True,
                service_context=ServiceContext.from_defaults(llm=self.llm, embed_model=self.embedded_model)
                )
            index.storage_context.persist(persist_dir=self.index_name)
        else:
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=self.index_name)
            )
        return index

    def index_pdf(self, pdf_path):
        canada_pdf = PDFReader().load_data(file=pdf_path)
        canada_index = self.get_index(canada_pdf)
        canada_engine = canada_index.as_query_engine()
        return canada_engine

# Usage:
# pdf_indexer = PDFIndexer("canada")
# canada_engine = pdf_indexer.index_pdf(os.path.join("data", "Canada.pdf"))


