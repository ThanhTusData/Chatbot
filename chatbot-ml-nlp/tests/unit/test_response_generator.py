import pytest
from src.response.response_generator import ResponseGenerator

def test_response_generator_basic():
    gen = ResponseGenerator()
    
    response = gen.generate('greeting', confidence=0.9)
    assert isinstance(response, str)
    assert len(response) > 0

def test_response_generator_low_confidence():
    gen = ResponseGenerator()
    
    response = gen.generate('unknown_intent', confidence=0.3)
    assert isinstance(response, str)

def test_response_with_context():
    gen = ResponseGenerator()
    gen.responses = {
        'product_inquiry': [
            'The price for {product} is {price}'
        ]
    }
    
    context = {'product': 'Widget', 'price': '$99'}
    response = gen.generate('product_inquiry', confidence=0.9, context=context)
    
    assert 'Widget' in response
    assert '$99' in response