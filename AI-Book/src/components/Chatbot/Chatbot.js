import React, { useState, useRef, useEffect } from 'react';
import './Chatbot.css';

const Chatbot = ({ backendUrl = 'https://your-hf-space-url.hf.space' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [userId, setUserId] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [userSessions, setUserSessions] = useState([]);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [showAuth, setShowAuth] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authToken, setAuthToken] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Check for saved auth token on component mount
  useEffect(() => {
    const savedToken = localStorage.getItem('chatbot_token');
    const savedUserId = localStorage.getItem('chatbot_user_id');
    if (savedToken && savedUserId) {
      setAuthToken(savedToken);
      setUserId(savedUserId);
    }
  }, []);

  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const authenticateUser = async () => {
    setIsAuthenticating(true);
    try {
      const response = await fetch(`${backendUrl}/auth`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: username,
          password: password
        })
      });

      if (!response.ok) {
        throw new Error(`Authentication failed: ${response.status}`);
      }

      const data = await response.json();
      setAuthToken(data.token);
      setUserId(data.user_id);
      
      // Save token to localStorage
      localStorage.setItem('chatbot_token', data.token);
      localStorage.setItem('chatbot_user_id', data.user_id);
      
      setShowAuth(false);
      setUsername('');
      setPassword('');
    } catch (error) {
      console.error('Authentication error:', error);
      alert('Authentication failed. Please try again.');
    } finally {
      setIsAuthenticating(false);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);

    // Add user message to UI immediately
    const userMsgObj = { role: 'user', content: userMessage, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsgObj]);

    try {
      // Get current page context
      const currentPath = window.location.pathname;
      const currentPageTitle = document.title;

      // Prepare the request to the backend
      const requestBody = {
        message: userMessage,
        context: `Current page: ${currentPageTitle} (${currentPath})`,
        user_id: userId
      };

      // Prepare headers with auth token if available
      const headers = {
        'Content-Type': 'application/json',
      };
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      // Call the backend API
      const response = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setSessionId(data.session_id); // Store session ID for continued conversation

      // Add bot response to UI
      const botMsgObj = { 
        role: 'assistant', 
        content: data.response, 
        timestamp: new Date().toISOString() 
      };
      setMessages(prev => [...prev, botMsgObj]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMsg = {
        role: 'assistant',
        content: `Sorry, I encountered an error processing your request. Please try again.`,
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  const loadUserHistory = async () => {
    if (!authToken) {
      alert('Please authenticate first to view history');
      setShowAuth(true);
      return;
    }

    try {
      const response = await fetch(`${backendUrl}/chat/history`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      });

      if (!response.ok) {
        if (response.status === 401) {
          // Token may have expired, clear it
          localStorage.removeItem('chatbot_token');
          localStorage.removeItem('chatbot_user_id');
          setAuthToken(null);
          setUserId(null);
          alert('Session expired. Please log in again.');
          setShowAuth(true);
        } else {
          throw new Error(`Server error: ${response.status}`);
        }
        return;
      }

      const data = await response.json();
      setUserSessions(data.sessions || []);
      setShowHistory(true);
    } catch (error) {
      console.error('Error loading history:', error);
      alert('Error loading history. Please try again.');
    }
  };

  const loadSessionMessages = (session) => {
    // Set the messages to the specific session's messages
    setMessages(session.messages.map(msg => ({
      ...msg,
      timestamp: new Date(msg.timestamp).toLocaleTimeString()
    })));
    setSessionId(session.session_id);
    setShowHistory(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  const handleLogout = () => {
    setAuthToken(null);
    setUserId(null);
    localStorage.removeItem('chatbot_token');
    localStorage.removeItem('chatbot_user_id');
    setMessages([]);
    setSessionId(null);
    setUserSessions([]);
    setShowHistory(false);
  };

  const quickAsk = (question) => {
    setInputValue(question);
    setTimeout(() => {
      sendMessage();
    }, 300);
  };

  return (
    <div className={`chatbot-container ${isOpen ? 'open' : ''}`}>
      {!isOpen ? (
        <button className="chatbot-fab" onClick={toggleChat} aria-label="Open chat">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="24px" height="24px">
            <path d="M0 0h24v24H0z" fill="none"/>
            <path d="M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
          </svg>
        </button>
      ) : (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="header-info">
              <h3>Physical AI Assistant</h3>
              {authToken && (
                <span className="user-info" title={`User: ${userId}`}>
                  {userId?.split('_')[1] || 'User'} 
                  <button className="logout-btn" onClick={handleLogout} title="Logout">
                    ×
                  </button>
                </span>
              )}
            </div>
            <div className="header-actions">
              {!authToken && !showAuth && (
                <button className="auth-btn" onClick={() => setShowAuth(true)} title="Login">
                  👤
                </button>
              )}
              <button 
                className="history-btn" 
                onClick={() => {
                  if (showHistory) {
                    setShowHistory(false);
                  } else {
                    loadUserHistory();
                  }
                }} 
                title="History"
              >
                📚
              </button>
              <button className="chatbot-close" onClick={toggleChat} aria-label="Close chat">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="white" width="18px" height="18px">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                  <path d="M0 0h24v24H0z" fill="none"/>
                </svg>
              </button>
            </div>
          </div>
          
          {showAuth ? (
            <div className="auth-container">
              <h4>Login to Access History</h4>
              <div className="auth-form">
                <input
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="auth-input"
                />
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="auth-input"
                />
                <div className="auth-buttons">
                  <button 
                    onClick={authenticateUser} 
                    disabled={isAuthenticating}
                    className="auth-submit-btn"
                  >
                    {isAuthenticating ? 'Logging in...' : 'Login'}
                  </button>
                  <button 
                    onClick={() => setShowAuth(false)}
                    className="auth-cancel-btn"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          ) : showHistory ? (
            <div className="history-container">
              <div className="history-header">
                <h4>Your Sessions</h4>
                <button 
                  onClick={() => setShowHistory(false)}
                  className="back-btn"
                >
                  ← Back to chat
                </button>
              </div>
              
              {userSessions.length === 0 ? (
                <p className="no-history">No conversation history available</p>
              ) : (
                <div className="sessions-list">
                  {userSessions.map((session, index) => (
                    <div 
                      key={session.session_id} 
                      className="session-item"
                      onClick={() => loadSessionMessages(session)}
                    >
                      <div className="session-header">
                        <strong>Session {userSessions.length - index}</strong>
                        <span className="session-date">
                          {new Date(session.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <div className="session-preview">
                        <span className="message-count">{session.message_count} messages</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="chatbot-messages">
                {messages.length === 0 ? (
                  <div className="chatbot-welcome">
                    <p>Hello! I'm your Physical AI & Humanoid Robotics assistant.</p>
                    <p>Ask me anything about the book content, and I'll do my best to help!</p>
                    <div className="suggested-questions">
                      <p><strong>Try asking about:</strong></p>
                      <div className="quick-ask-buttons">
                        <button onClick={() => quickAsk("What is Physical AI?")}>Physical AI</button>
                        <button onClick={() => quickAsk("Explain ROS 2 architecture")}>ROS 2</button>
                        <button onClick={() => quickAsk("How do humanoid robots navigate?")}>Navigation</button>
                        <button onClick={() => quickAsk("What are digital twins in robotics?")}>Digital Twins</button>
                        <button onClick={() => quickAsk("Explain humanoid locomotion")}>Locomotion</button>
                        <button onClick={() => quickAsk("How do VLA models work?")}>VLA Models</button>
                      </div>
                    </div>
                  </div>
                ) : (
                  messages.map((msg, index) => (
                    <div 
                      key={index} 
                      className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}
                    >
                      <div className="message-content">
                        {msg.content.split('\n').map((line, i) => (
                          <p key={i}>{line}</p>
                        ))}
                      </div>
                      <div className="message-timestamp">
                        {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
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
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about Physical AI, Robotics, or book content..."
                  rows="1"
                  disabled={isLoading}
                />
                <button 
                  onClick={sendMessage} 
                  disabled={!inputValue.trim() || isLoading}
                  className="send-button"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20px" height="20px">
                    <path d="M0 0h24v24H0z" fill="none"/>
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                  </svg>
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Chatbot;