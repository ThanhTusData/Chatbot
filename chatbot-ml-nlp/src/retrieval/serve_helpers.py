from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import hashlib
import json

class ServingHelpers:
    """Helper functions for serving layer"""
    
    @staticmethod
    def format_search_results(
        results: List[Dict],
        include_metadata: bool = True,
        max_content_length: int = 500
    ) -> List[Dict]:
        """Format search results for API response"""
        formatted_results = []
        
        for result in results:
            formatted = {
                'content': result.get('content', '')[:max_content_length],
                'score': round(result.get('score', 0.0), 4)
            }
            
            if include_metadata and 'metadata' in result:
                formatted['metadata'] = result['metadata']
            
            # Add snippet with highlighting
            if len(result.get('content', '')) > max_content_length:
                formatted['content'] += '...'
            
            formatted_results.append(formatted)
        
        return formatted_results
    
    @staticmethod
    def generate_session_id(user_id: Optional[str] = None) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        data = f"{user_id or 'anonymous'}_{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()
    
    @staticmethod
    def validate_query(query: str, min_length: int = 1, max_length: int = 1000) -> tuple:
        """Validate search query"""
        if not query:
            return False, "Query cannot be empty"
        
        if len(query) < min_length:
            return False, f"Query must be at least {min_length} characters"
        
        if len(query) > max_length:
            return False, f"Query must not exceed {max_length} characters"
        
        return True, "Valid"
    
    @staticmethod
    def calculate_confidence_level(confidence: float) -> str:
        """Convert confidence score to level"""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    @staticmethod
    def merge_search_results(
        intent_results: List[Dict],
        semantic_results: List[Dict],
        intent_weight: float = 0.4,
        semantic_weight: float = 0.6
    ) -> List[Dict]:
        """Merge and rank results from multiple sources"""
        merged = {}
        
        # Add intent-based results
        for result in intent_results:
            doc_id = result.get('id', result.get('content', '')[:50])
            merged[doc_id] = {
                **result,
                'combined_score': result.get('confidence', 0.0) * intent_weight
            }
        
        # Add semantic search results
        for result in semantic_results:
            doc_id = result.get('id', result.get('content', '')[:50])
            if doc_id in merged:
                merged[doc_id]['combined_score'] += result.get('score', 0.0) * semantic_weight
            else:
                merged[doc_id] = {
                    **result,
                    'combined_score': result.get('score', 0.0) * semantic_weight
                }
        
        # Sort by combined score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_results
    
    @staticmethod
    def create_response_metadata(
        processing_time: float,
        model_version: str = "1.0.0",
        additional_info: Optional[Dict] = None
    ) -> Dict:
        """Create response metadata"""
        metadata = {
            'processing_time_ms': round(processing_time * 1000, 2),
            'model_version': model_version,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        return metadata
    
    @staticmethod
    def filter_results_by_threshold(
        results: List[Dict],
        threshold: float = 0.7,
        score_key: str = 'score'
    ) -> List[Dict]:
        """Filter results by minimum score threshold"""
        return [
            result for result in results
            if result.get(score_key, 0.0) >= threshold
        ]
    
    @staticmethod
    def deduplicate_results(
        results: List[Dict],
        similarity_threshold: float = 0.95
    ) -> List[Dict]:
        """Remove duplicate or very similar results"""
        unique_results = []
        seen_contents = []
        
        for result in results:
            content = result.get('content', '')
            
            # Simple similarity check
            is_duplicate = False
            for seen_content in seen_contents:
                if len(content) > 0 and len(seen_content) > 0:
                    # Simple Jaccard similarity
                    set1 = set(content.lower().split())
                    set2 = set(seen_content.lower().split())
                    similarity = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                    
                    if similarity >= similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_contents.append(content)
        
        return unique_results
    
    @staticmethod
    def prepare_context_window(
        conversation_history: List[Dict],
        max_messages: int = 10,
        max_tokens: int = 2000
    ) -> List[Dict]:
        """Prepare conversation context window"""
        # Take last N messages
        recent_messages = conversation_history[-max_messages:]
        
        # Estimate tokens (simple: 1 token ≈ 4 characters)
        total_chars = sum(len(msg.get('text', '')) for msg in recent_messages)
        
        # Truncate if exceeds max tokens
        if total_chars > max_tokens * 4:
            truncated = []
            current_chars = 0
            
            for msg in reversed(recent_messages):
                msg_chars = len(msg.get('text', ''))
                if current_chars + msg_chars <= max_tokens * 4:
                    truncated.insert(0, msg)
                    current_chars += msg_chars
                else:
                    break
            
            return truncated
        
        return recent_messages
    
    @staticmethod
    def create_cache_key(
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> str:
        """Create cache key for query results"""
        cache_data = {
            'query': query.lower().strip(),
            'filters': filters or {},
            'top_k': top_k
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    @staticmethod
    def extract_entities_from_query(query: str) -> Dict[str, List[str]]:
        """Extract potential entities from query"""
        entities = {
            'numbers': [],
            'emails': [],
            'urls': []
        }
        
        import re
        
        # Extract numbers
        numbers = re.findall(r'\b\d+\b', query)
        entities['numbers'] = numbers
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        entities['emails'] = emails
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', query)
        entities['urls'] = urls
        
        return entities
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input"""
        import html
        
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def format_error_response(
        error_code: str,
        error_message: str,
        details: Optional[Dict] = None
    ) -> Dict:
        """Format error response"""
        response = {
            'error': {
                'code': error_code,
                'message': error_message,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if details:
            response['error']['details'] = details
        
        return response


# Utility functions that can be imported directly
def quick_validate(query: str) -> bool:
    """Quick validation for query"""
    is_valid, _ = ServingHelpers.validate_query(query)
    return is_valid


def get_confidence_level(score: float) -> str:
    """Get confidence level from score"""
    return ServingHelpers.calculate_confidence_level(score)


# Example usage
if __name__ == "__main__":
    # Test helpers
    helper = ServingHelpers()
    
    # Test session ID generation
    session_id = helper.generate_session_id("user123")
    print(f"Session ID: {session_id}")
    
    # Test query validation
    is_valid, message = helper.validate_query("test query")
    print(f"Valid: {is_valid}, Message: {message}")
    
    # Test confidence level
    level = helper.calculate_confidence_level(0.85)
    print(f"Confidence Level: {level}")