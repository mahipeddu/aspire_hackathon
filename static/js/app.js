// Modern WebSocket Chat Application

class ChatApp {
    constructor() {
        this.ws = null;
        this.clientId = this.generateClientId();
        this.currentModel = 'llama2-7b-8bit';
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connect();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.elements = {
            messages: document.getElementById('messages'),
            messageInput: document.getElementById('message-input'),
            sendBtn: document.getElementById('send-btn'),
            modelSelect: document.getElementById('model-select'),
            clearBtn: document.getElementById('clear-btn'),
            clearMemoryBtn: document.getElementById('clear-memory-btn'),
            memoryInfo: document.getElementById('memory-info'),
            memoryText: document.getElementById('memory-text'),
            typing: document.getElementById('typing'),
            typingText: document.getElementById('typing-text'),
            statusText: document.getElementById('status-text'),
            statusIndicator: document.getElementById('status-indicator'),
            stats: document.getElementById('stats'),
            statsText: document.getElementById('stats-text')
        };
    }
    
    setupEventListeners() {
        // Send button click
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // Enter key to send (Shift+Enter for new line)
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        this.elements.messageInput.addEventListener('input', () => this.resizeTextarea());
        
        // Model selection change
        this.elements.modelSelect.addEventListener('change', () => this.switchModel());
        
        // Clear conversation
        this.elements.clearBtn.addEventListener('click', () => this.clearConversation());
        
        // Clear memory
        this.elements.clearMemoryBtn.addEventListener('click', () => this.clearMemory());
        
        // Window focus to reconnect if needed
        window.addEventListener('focus', () => {
            if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
                this.connect();
            }
        });
    }
    
    connect() {
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws/${this.clientId}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => this.onConnect();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onDisconnect();
            this.ws.onerror = (error) => this.onError(error);
            
        } catch (error) {
            console.error('Error connecting to WebSocket:', error);
            this.updateConnectionStatus('disconnected', 'Connection failed');
            this.scheduleReconnect();
        }
    }
    
    onConnect() {
        console.log('Connected to chat server');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus('connected', 'Connected');
        this.elements.sendBtn.disabled = false;
        this.elements.messageInput.disabled = false;
        
        // Update memory info when connected
        this.updateMemoryInfo();
    }
    
    onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }
    
    onDisconnect() {
        console.log('Disconnected from chat server');
        this.updateConnectionStatus('disconnected', 'Disconnected');
        this.elements.sendBtn.disabled = true;
        this.hideTyping();
        this.scheduleReconnect();
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
        this.updateConnectionStatus('disconnected', 'Connection error');
    }
    
    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
            
            this.updateConnectionStatus('connecting', `Reconnecting in ${delay/1000}s...`);
            
            setTimeout(() => {
                this.connect();
            }, delay);
        } else {
            this.updateConnectionStatus('disconnected', 'Connection failed. Please refresh.');
        }
    }
    
    updateConnectionStatus(status, text) {
        this.elements.statusText.textContent = text;
        this.elements.statusIndicator.className = `status-indicator ${status}`;
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'response':
                this.hideTyping();
                this.addMessage('assistant', data.message, data.model, data.stats);
                // Update memory info after response to show current usage
                setTimeout(() => this.updateMemoryInfo(), 1000);
                break;
                
            case 'typing':
                this.showTyping(data.message);
                break;
                
            case 'status':
                this.showStatus(data.message, 'info');
                break;
                
            case 'error':
                this.hideTyping();
                this.showStatus(data.message, 'error');
                break;
                
            case 'model_switched':
                this.currentModel = data.model;
                this.showStatus(data.message, 'success');
                if (data.memory_info) {
                    this.updateMemoryInfo({ memory_usage: data.memory_info });
                }
                break;
                
            case 'history':
                this.loadConversationHistory(data.messages);
                break;
                
            case 'history_cleared':
                this.clearMessagesUI();
                this.showStatus(data.message, 'success');
                break;
        }
    }
    
    sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }
        
        // Add user message to UI immediately
        this.addMessage('user', message);
        
        // Clear input
        this.elements.messageInput.value = '';
        this.resizeTextarea();
        
        // Send to server
        this.ws.send(JSON.stringify({
            type: 'chat',
            message: message,
            model: this.currentModel
        }));
    }
    
    switchModel() {
        const newModel = this.elements.modelSelect.value;
        if (newModel === this.currentModel) {
            return;
        }
        
        this.currentModel = newModel;
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'model_switch',
                model: newModel
            }));
        }
    }
    
    clearConversation() {
        if (confirm('Are you sure you want to clear the conversation history?')) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'clear_history'
                }));
            }
        }
    }
    
    addMessage(role, content, model = null, stats = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        const metaDiv = document.createElement('div');
        metaDiv.className = 'message-meta';
        
        if (role === 'assistant' && model) {
            const modelBadge = document.createElement('span');
            modelBadge.className = 'model-badge';
            modelBadge.textContent = model.replace('llama2-7b-', '').toUpperCase();
            metaDiv.appendChild(modelBadge);
            
            if (stats && stats.speed) {
                const statsBadge = document.createElement('span');
                statsBadge.className = 'stats-badge';
                statsBadge.textContent = `${stats.speed.toFixed(1)} tok/s`;
                metaDiv.appendChild(statsBadge);
                
                // Show detailed stats briefly
                this.showStats(`Generated ${stats.tokens} tokens in ${stats.time.toFixed(2)}s at ${stats.speed.toFixed(1)} tokens/sec`);
            }
        }
        
        const timestamp = document.createElement('span');
        timestamp.textContent = new Date().toLocaleTimeString();
        metaDiv.appendChild(timestamp);
        
        messageDiv.appendChild(contentDiv);
        if (metaDiv.children.length > 0) {
            messageDiv.appendChild(metaDiv);
        }
        
        // Remove welcome message when first real message is added
        const welcomeMsg = this.elements.messages.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }
        
        this.elements.messages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTyping(text = 'AI is thinking...') {
        this.elements.typingText.textContent = text;
        this.elements.typing.style.display = 'block';
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.elements.typing.style.display = 'none';
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.createElement('div');
        statusDiv.className = `status-message ${type}`;
        statusDiv.textContent = message;
        
        this.elements.messages.appendChild(statusDiv);
        this.scrollToBottom();
        
        // Auto-remove status messages after 3 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.remove();
            }
        }, 3000);
    }
    
    showStats(text) {
        this.elements.statsText.textContent = text;
        this.elements.stats.style.display = 'block';
        
        // Hide stats after 5 seconds
        setTimeout(() => {
            this.elements.stats.style.display = 'none';
        }, 5000);
    }
    
    loadConversationHistory(messages) {
        // Clear existing messages except welcome
        const messageElements = this.elements.messages.querySelectorAll('.message:not(.welcome-message)');
        messageElements.forEach(el => el.remove());
        
        // Add historical messages
        messages.forEach(msg => {
            if (msg.role === 'user' || msg.role === 'assistant') {
                this.addMessage(msg.role, msg.content, msg.model);
            }
        });
    }
    
    clearMessagesUI() {
        // Remove all messages except welcome
        const messageElements = this.elements.messages.querySelectorAll('.message:not(.welcome-message), .status-message');
        messageElements.forEach(el => el.remove());
        
        // Show welcome message again
        const welcomeMsg = this.elements.messages.querySelector('.welcome-message');
        if (!welcomeMsg) {
            location.reload(); // Reload to show welcome message
        }
    }
    
    resizeTextarea() {
        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
        }, 100);
    }
    
    async clearMemory() {
        try {
            this.showStatus('Clearing GPU memory...', 'info');
            this.elements.clearMemoryBtn.disabled = true;
            
            const response = await fetch('/api/memory/clear', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showStatus(data.message, 'success');
                this.updateMemoryInfo(data.memory_usage);
            } else {
                this.showStatus(data.message, 'error');
            }
            
        } catch (error) {
            console.error('Error clearing memory:', error);
            this.showStatus('Failed to clear memory', 'error');
        } finally {
            this.elements.clearMemoryBtn.disabled = false;
        }
    }
    
    async updateMemoryInfo(memoryData = null) {
        try {
            if (!memoryData) {
                const response = await fetch('/api/memory');
                memoryData = await response.json();
            }
            
            let memoryText = 'Memory: ';
            if (memoryData.memory_usage && memoryData.memory_usage.gpu_0) {
                const gpu = memoryData.memory_usage.gpu_0;
                const utilization = Math.round(gpu.utilization_percent || 0);
                const reserved = gpu.reserved_gb.toFixed(1);
                const total = gpu.total_gb.toFixed(1);
                memoryText = `GPU: ${reserved}/${total}GB (${utilization}%)`;
            } else {
                memoryText = 'CPU Mode';
            }
            
            if (memoryData.loaded_models_count) {
                memoryText += ` | ${memoryData.loaded_models_count} models`;
            }
            
            this.elements.memoryText.textContent = memoryText;
            
        } catch (error) {
            console.error('Error fetching memory info:', error);
            this.elements.memoryText.textContent = 'Memory: Error';
        }
    }
}

// Initialize the chat application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});

// Handle page visibility for connection management
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible' && window.chatApp) {
        if (!window.chatApp.ws || window.chatApp.ws.readyState === WebSocket.CLOSED) {
            window.chatApp.connect();
        }
    }
});
