import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const Chatbot = ({ backendUrl = 'http://localhost:8000' }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Function to get selected text from the page
  const getSelectedText = () => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText) {
      setSelectedText(selectedText);
      // Show a notification to the user
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
      <div className="chatbot-header">
        <h3>AI Assistant for Physical AI & Humanoid Robotics</h3>
        <div className="header-buttons">
          <button onClick={getSelectedText} className="btn-secondary" title="Use selected text from page">
            Use Selected Text
          </button>
          <button onClick={clearChat} className="btn-secondary">
            Clear Chat
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
          <div className="welcome-message">
            <h4>Hello! I'm your AI assistant for the Physical AI & Humanoid Robotics book.</h4>
            <p>Ask me questions about the book content, or:</p>
            <ul>
              <li>Select text on the page and click "Use Selected Text" to ask about specific content</li>
              <li>Ask general questions about Physical AI, ROS, Humanoid Robotics, etc.</li>
            </ul>
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
        <div ref={messagesEndRef} />
      </div>

      <div className="chatbot-input-area">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about Physical AI & Humanoid Robotics..."
          rows="3"
          disabled={isLoading}
          className="chat-input"
        />
        <button
          onClick={sendMessage}
          disabled={!inputValue.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default Chatbot;