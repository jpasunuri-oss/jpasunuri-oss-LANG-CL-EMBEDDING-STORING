import os
import unittest

from langchain.vectorstores import FAISS
from langchain_core.outputs import LLMResult

from src.main.app import get_pdf_text, get_text_chunks, get_vector_store
from src.utilities.llm_testing_util import llm_connection_check, llm_wakeup, classify_relevancy


class TestLangChainLab(unittest.TestCase):

    """
       This function is a sanity check for the Language Learning Model (LLM) connection.
       It attempts to generate a response from the LLM. If a 'Bad Gateway' error is encountered,
       it initiates the LLM wake-up process. This function is critical for ensuring the LLM is
       operational before running tests and should not be modified without understanding the
       implications.
       Raises:
           Exception: If any error other than 'Bad Gateway' is encountered, it is raised to the caller.
       """

    def test_llm_sanity_check(self):
        try:
            response = llm_connection_check()
            self.assertIsInstance(response, LLMResult)
        except Exception as e:
            if 'Bad Gateway' in str(e):
                llm_wakeup()
                self.fail("LLM is not awake. Please try again in 3-5 minutes.")

    def test_get_pdf_text(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, "..", "..")
        folder_path = os.path.join(root_dir, "resources")
        test_pdf_path = os.path.join(folder_path, "langchain.pdf")
        text = get_pdf_text(test_pdf_path)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_get_pdf_text_with_invalid_path(self):
        invalid_pdf_path = "non_existent_path/non_existent_file.pdf"
        with self.assertRaises(FileNotFoundError):
            get_pdf_text(invalid_pdf_path)

    def test_get_text_chunks(self):
        sample_text = "This is a test. " * 100
        chunks = get_text_chunks(sample_text)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertTrue(len(chunks) > 0)

    def test_get_vector_store(self):
        sample_chunks = ["This is a test.", "Another test sentence."]
        vector_store = get_vector_store(sample_chunks)
        self.assertIsInstance(vector_store, FAISS)


if __name__ == "__main__":
    unittest.main()
