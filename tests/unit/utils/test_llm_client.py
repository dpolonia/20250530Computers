"""
Unit tests for the LLM client utilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from src.utils.llm_client import (
    BaseLLMClient,
    AnthropicClient,
    OpenAIClient,
    GeminiClient,
    get_llm_client
)


class TestBaseLLMClient:
    """Tests for the BaseLLMClient class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        # Arrange
        client = BaseLLMClient(api_key="fake_key")
        
        # Act/Assert
        with pytest.raises(NotImplementedError):
            client.validate_api_key()
        
        with pytest.raises(NotImplementedError):
            client._get_completion_without_verification("test prompt")
    
    def test_get_completion_calls_verify(self):
        """Test that get_completion calls verify_model_accuracy."""
        # Arrange
        client = BaseLLMClient(api_key="fake_key")
        
        # Mock the abstract methods
        client.validate_api_key = Mock(return_value=True)
        client._get_completion_without_verification = Mock(return_value="test response")
        client.verify_model_accuracy = Mock(return_value=True)
        
        # Act
        response = client.get_completion("test prompt")
        
        # Assert
        assert response == "test response"
        client.verify_model_accuracy.assert_called_once()
        client._get_completion_without_verification.assert_called_once_with(
            "test prompt", None, None, False
        )
    
    def test_get_completion_skips_verification_if_verified(self):
        """Test that get_completion skips verification if already verified."""
        # Arrange
        client = BaseLLMClient(api_key="fake_key")
        
        # Mock the abstract methods
        client.validate_api_key = Mock(return_value=True)
        client._get_completion_without_verification = Mock(return_value="test response")
        client.verify_model_accuracy = Mock(return_value=True)
        
        # Set as already verified
        client.verified = True
        
        # Act
        response = client.get_completion("test prompt")
        
        # Assert
        assert response == "test response"
        client.verify_model_accuracy.assert_not_called()
        client._get_completion_without_verification.assert_called_once()
    
    def test_get_completion_force_verification(self):
        """Test that get_completion forces verification when requested."""
        # Arrange
        client = BaseLLMClient(api_key="fake_key")
        
        # Mock the abstract methods
        client.validate_api_key = Mock(return_value=True)
        client._get_completion_without_verification = Mock(return_value="test response")
        client.verify_model_accuracy = Mock(return_value=True)
        
        # Set as already verified
        client.verified = True
        
        # Act
        response = client.get_completion("test prompt", force_verification=True)
        
        # Assert
        assert response == "test response"
        client.verify_model_accuracy.assert_called_once()
        client._get_completion_without_verification.assert_called_once()
    
    def test_get_usage_statistics(self):
        """Test getting usage statistics."""
        # Arrange
        client = BaseLLMClient(api_key="fake_key")
        
        # Set some usage stats
        client.usage_stats = {
            "total_tokens": 1000,
            "input_tokens": 700,
            "output_tokens": 300,
            "total_cost": 0.02,
            "calls": 5
        }
        
        # Act
        stats = client.get_usage_statistics()
        
        # Assert
        assert stats["total_tokens"] == 1000
        assert stats["input_tokens"] == 700
        assert stats["output_tokens"] == 300
        assert stats["total_cost"] == 0.02
        assert stats["calls"] == 5
        assert "summary" in stats


class TestAnthropicClient:
    """Tests for the AnthropicClient class."""
    
    @patch('anthropic.Anthropic')
    def test_initialization(self, mock_anthropic_class):
        """Test client initialization."""
        # Arrange
        mock_anthropic_instance = Mock()
        mock_anthropic_class.return_value = mock_anthropic_instance
        
        # Act
        client = AnthropicClient(api_key="fake_key")
        
        # Assert
        mock_anthropic_class.assert_called_once_with(api_key="fake_key")
        assert client.client is mock_anthropic_instance
        assert client.provider == "anthropic"
    
    @patch('anthropic.Anthropic')
    def test_validate_api_key(self, mock_anthropic_class):
        """Test API key validation."""
        # Arrange
        mock_anthropic_instance = Mock()
        mock_anthropic_class.return_value = mock_anthropic_instance
        mock_anthropic_instance.models.list.return_value = ["model1", "model2"]
        
        client = AnthropicClient(api_key="fake_key")
        
        # Act
        is_valid = client.validate_api_key()
        
        # Assert
        assert is_valid is True
        mock_anthropic_instance.models.list.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_validate_api_key_fails(self, mock_anthropic_class):
        """Test API key validation failure."""
        # Arrange
        mock_anthropic_instance = Mock()
        mock_anthropic_class.return_value = mock_anthropic_instance
        mock_anthropic_instance.models.list.side_effect = Exception("Invalid API key")
        
        client = AnthropicClient(api_key="fake_key")
        
        # Act
        is_valid = client.validate_api_key()
        
        # Assert
        assert is_valid is False
        mock_anthropic_instance.models.list.assert_called_once()
    
    @patch('anthropic.Anthropic')
    def test_get_completion(self, mock_anthropic_class):
        """Test getting a completion."""
        # Arrange
        mock_anthropic_instance = Mock()
        mock_anthropic_class.return_value = mock_anthropic_instance
        
        mock_response = Mock()
        mock_response.content = [{"type": "text", "text": "test response"}]
        mock_response.usage = {"input_tokens": 10, "output_tokens": 5}
        
        mock_anthropic_instance.messages.create.return_value = mock_response
        
        client = AnthropicClient(api_key="fake_key", model="claude-3-opus-20240229")
        client.verified = True  # Skip verification
        
        # Act
        response = client.get_completion("test prompt", "test system prompt")
        
        # Assert
        assert response == "test response"
        mock_anthropic_instance.messages.create.assert_called_once()
        assert client.usage_stats["input_tokens"] == 10
        assert client.usage_stats["output_tokens"] == 5


class TestOpenAIClient:
    """Tests for the OpenAIClient class."""
    
    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai_class):
        """Test client initialization."""
        # Arrange
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        
        # Act
        client = OpenAIClient(api_key="fake_key")
        
        # Assert
        mock_openai_class.assert_called_once_with(api_key="fake_key")
        assert client.client is mock_openai_instance
        assert client.provider == "openai"
    
    @patch('openai.OpenAI')
    def test_validate_api_key(self, mock_openai_class):
        """Test API key validation."""
        # Arrange
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        mock_openai_instance.models.list.return_value.data = ["model1", "model2"]
        
        client = OpenAIClient(api_key="fake_key")
        
        # Act
        is_valid = client.validate_api_key()
        
        # Assert
        assert is_valid is True
        mock_openai_instance.models.list.assert_called_once()
    
    @patch('openai.OpenAI')
    def test_get_completion(self, mock_openai_class):
        """Test getting a completion."""
        # Arrange
        mock_openai_instance = Mock()
        mock_openai_class.return_value = mock_openai_instance
        
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="test response"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        client = OpenAIClient(api_key="fake_key", model="gpt-4")
        client.verified = True  # Skip verification
        
        # Act
        response = client.get_completion("test prompt", "test system prompt")
        
        # Assert
        assert response == "test response"
        mock_openai_instance.chat.completions.create.assert_called_once()
        assert client.usage_stats["input_tokens"] == 10
        assert client.usage_stats["output_tokens"] == 5


class TestGeminiClient:
    """Tests for the GeminiClient class."""
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_initialization(self, mock_model_class, mock_configure):
        """Test client initialization."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        # Act
        client = GeminiClient(api_key="fake_key")
        
        # Assert
        mock_configure.assert_called_once_with(api_key="fake_key")
        mock_model_class.assert_called_once()
        assert client.model_instance is mock_model_instance
        assert client.provider == "google"
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.list_models')
    def test_validate_api_key(self, mock_list_models, mock_configure):
        """Test API key validation."""
        # Arrange
        mock_list_models.return_value = ["model1", "model2"]
        
        client = GeminiClient(api_key="fake_key")
        
        # Act
        is_valid = client.validate_api_key()
        
        # Assert
        assert is_valid is True
        mock_list_models.assert_called_once()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_completion(self, mock_model_class, mock_configure):
        """Test getting a completion."""
        # Arrange
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance
        
        mock_response = Mock()
        mock_response.text = "test response"
        mock_response.prompt_token_count = 10
        mock_response.candidates = [Mock(token_count=5)]
        
        mock_model_instance.generate_content.return_value = mock_response
        
        client = GeminiClient(api_key="fake_key", model="gemini-1.5-pro")
        client.verified = True  # Skip verification
        
        # Act
        response = client.get_completion("test prompt", "test system prompt")
        
        # Assert
        assert response == "test response"
        mock_model_instance.generate_content.assert_called_once()
        assert client.usage_stats["input_tokens"] == 10
        assert client.usage_stats["output_tokens"] == 5


class TestGetLLMClient:
    """Tests for the get_llm_client function."""
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_anthropic_key"})
    @patch('src.utils.llm_client.AnthropicClient')
    def test_get_anthropic_client(self, mock_anthropic_client):
        """Test getting an Anthropic client."""
        # Arrange
        mock_instance = Mock()
        mock_anthropic_client.return_value = mock_instance
        
        # Act
        client = get_llm_client("anthropic", "claude-3-opus-20240229")
        
        # Assert
        assert client is mock_instance
        mock_anthropic_client.assert_called_once_with(
            api_key="fake_anthropic_key", 
            model="claude-3-opus-20240229"
        )
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_openai_key"})
    @patch('src.utils.llm_client.OpenAIClient')
    def test_get_openai_client(self, mock_openai_client):
        """Test getting an OpenAI client."""
        # Arrange
        mock_instance = Mock()
        mock_openai_client.return_value = mock_instance
        
        # Act
        client = get_llm_client("openai", "gpt-4")
        
        # Assert
        assert client is mock_instance
        mock_openai_client.assert_called_once_with(
            api_key="fake_openai_key", 
            model="gpt-4"
        )
    
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_google_key"})
    @patch('src.utils.llm_client.GeminiClient')
    def test_get_google_client(self, mock_google_client):
        """Test getting a Google client."""
        # Arrange
        mock_instance = Mock()
        mock_google_client.return_value = mock_instance
        
        # Act
        client = get_llm_client("google", "gemini-1.5-pro")
        
        # Assert
        assert client is mock_instance
        mock_google_client.assert_called_once_with(
            api_key="fake_google_key", 
            model="gemini-1.5-pro"
        )
    
    def test_invalid_provider(self):
        """Test getting a client with an invalid provider."""
        # Act/Assert
        with pytest.raises(ValueError):
            get_llm_client("invalid_provider", "model")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test getting a client with missing API key."""
        # Act/Assert
        with pytest.raises(ValueError):
            get_llm_client("anthropic", "claude-3-opus-20240229")