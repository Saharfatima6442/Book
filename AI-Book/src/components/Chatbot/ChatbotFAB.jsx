import React, { useState, useEffect } from 'react';
import './ChatbotFAB.css';

const ChatbotFAB = ({ backendUrl = 'http://localhost:8000' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');

  // Function to get selected text from the page
  const getSelectedText = () => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText) {
      setSelectedText(selectedText);
      alert(`Selected text captured: "${selectedText.substring(0, 50)}${selectedText.length > 50 ? '...' : ''}"\n\nYou can now ask questions about this text.`);
    } else {
      alert('Please select text on the page first, then ask your question.');
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = { role: 'user', content: inputValue, timestamp: new Date() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare the request body
      const requestBody = {
        message: inputValue,
        selected_text: selectedText || null
      };

      const response = await fetch(`${backendUrl}/api/rag-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
        tokens_used: data.tokens_used,
        timestamp: new Date(),
      };

      setMessages([...newMessages, botMessage]);
      // Clear selected text after use
      if (selectedText) {
        setSelectedText('');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedText('');
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <h3>AI Assistant</h3>
            <div className="header-actions">
              <button onClick={getSelectedText} className="auth-btn" title="Use selected text">
                📝
              </button>
              <button onClick={clearChat} className="auth-btn" title="Clear chat">
                🗑️
              </button>
              <button onClick={() => setIsOpen(false)} className="chatbot-close">
                ✕
              </button>
            </div>
          </div>

          {selectedText && (
            <div className="selected-text-preview">
              <strong>Context:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"
              <button onClick={() => setSelectedText('')} className="remove-btn">×</button>
            </div>
          )}

          <div className="chatbot-messages">
            {messages.length === 0 ? (
              <div className="chatbot-welcome">
                <p>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics book.</p>
                <p>Ask me questions about the book content!</p>
                <div className="suggested-questions">
                  <p>Try asking:</p>
                  <div className="quick-ask-buttons">
                    <button onClick={() => setInputValue('What is Physical AI?')}>What is Physical AI?</button>
                    <button onClick={() => setInputValue('Explain ROS2 in robotics')}>Explain ROS2 in robotics</button>
                    <button onClick={() => setInputValue('How do humanoid robots maintain balance?')}>How do humanoid robots maintain balance?</button>
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-content">
                    {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        Sources: {message.sources.join(', ')}
                      </div>
                    )}
                    <div>{message.content}</div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="message assistant">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="chatbot-input-area">
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about Physical AI & Humanoid Robotics..."
              rows="1"
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="send-button"
            >
              ➤
            </button>
          </div>
        </div>
      ) : (
        <button className="chatbot-fab" onClick={() => setIsOpen(true)}>
          🤖
        </button>
      )}
    </div>
  );
};

export default ChatbotFAB;