import unittest
from config import Config
from src.core.chatbot import IntelligentChatbot


def create_test_suite():
    """Tạo test suite cho chatbot"""
    
    class TestChatbot(unittest.TestCase):
        
        def setUp(self):
            self.config = Config()
            self.chatbot = IntelligentChatbot(self.config)
            
        def test_nlp_processing(self):
            """Test NLP preprocessing"""
            text = "Hello, how are you today?"
            result = self.chatbot.nlp_processor.preprocess_text(text)
            
            self.assertIn('lemmatized', result)
            self.assertIn('entities', result)
            self.assertIn('processed_text', result)
            
        def test_intent_prediction(self):
            """Test intent classification"""
            # This would require a trained model
            pass
            
        def test_response_generation(self):
            """Test response generation"""
            response = self.chatbot.response_generator.generate_response(
                'greeting', 0.9, 'hello'
            )
            
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
            
        def test_conversation_history(self):
            """Test conversation history"""
            history = self.chatbot.conversation_history
            history.add_message("Hello", "Hi there", "greeting", 0.9)
            
            self.assertEqual(len(history.conversations), 1)
            self.assertEqual(history.conversations[0]['user_message'], "Hello")
            
    return TestChatbot