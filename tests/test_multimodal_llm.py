import pytest
from unittest.mock import Mock, patch

from livekit.agents import llm
from livekit.agents.types import NOT_GIVEN


def test_multimodal_llm_import():
    """Test that MultimodalLLM can be imported from OpenAI plugin"""
    try:
        from livekit.plugins.openai import MultimodalLLM
        assert MultimodalLLM is not None
    except ImportError:
        pytest.skip("OpenAI plugin not available")


def test_multimodal_llm_instantiation():
    """Test that MultimodalLLM can be instantiated with default parameters"""
    try:
        from livekit.plugins.openai import MultimodalLLM
        
        with patch('openai.AsyncClient'):
            llm_instance = MultimodalLLM(
                model="gpt-4o-audio-preview",
                enable_multimodal=True
            )
            
            assert llm_instance.enable_multimodal is True
            assert llm_instance.model == "gpt-4o-audio-preview"
            
    except ImportError:
        pytest.skip("OpenAI plugin not available")


def test_multimodal_llm_disable_flag():
    """Test that MultimodalLLM can be disabled via flag"""
    try:
        from livekit.plugins.openai import MultimodalLLM
        
        with patch('openai.AsyncClient'):
            llm_instance = MultimodalLLM(
                model="gpt-4o-audio-preview",
                enable_multimodal=False
            )
            
            assert llm_instance.enable_multimodal is False
            
    except ImportError:
        pytest.skip("OpenAI plugin not available")


def test_multimodal_llm_chat_method_signature():
    """Test that MultimodalLLM chat method accepts audio_data parameter"""
    try:
        from livekit.plugins.openai import MultimodalLLM
        
        with patch('openai.AsyncClient'):
            llm_instance = MultimodalLLM(enable_multimodal=True)
            
            chat_ctx = llm.ChatContext().append(text="Hello", role="user")
            
            with patch.object(llm_instance.__class__.__bases__[0], 'chat') as mock_chat:
                mock_stream = Mock()
                mock_chat.return_value = mock_stream
                
                result = llm_instance.chat(
                    chat_ctx=chat_ctx,
                    audio_data=b"fake_audio_data"
                )
                
                assert mock_chat.called
                assert result == mock_stream
                
    except ImportError:
        pytest.skip("OpenAI plugin not available")


def test_multimodal_llm_audio_data_not_given():
    """Test that MultimodalLLM works normally when audio_data is not provided"""
    try:
        from livekit.plugins.openai import MultimodalLLM
        
        with patch('openai.AsyncClient'):
            llm_instance = MultimodalLLM(enable_multimodal=True)
            
            chat_ctx = llm.ChatContext().append(text="Hello", role="user")
            
            with patch.object(llm_instance.__class__.__bases__[0], 'chat') as mock_chat:
                mock_stream = Mock()
                mock_chat.return_value = mock_stream
                
                result = llm_instance.chat(chat_ctx=chat_ctx)
                
                assert mock_chat.called
                assert result == mock_stream
                
    except ImportError:
        pytest.skip("OpenAI plugin not available")
