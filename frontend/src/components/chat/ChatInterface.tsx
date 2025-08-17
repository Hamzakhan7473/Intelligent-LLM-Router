'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'

interface Message {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  routing_info?: {
    model: string
    provider: string
    routing_decision: string
    response_time_ms: number
    usage: {
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
    }
  }
}

interface LLMRouterProps {
  className?: string
}

export default function LLMRouterInterface({ className }: LLMRouterProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [taskType, setTaskType] = useState<string>('chat')
  const [forceModel, setForceModel] = useState<string>('')
  const [preferSpeed, setPreferSpeed] = useState(false)
  const [preferQuality, setPreferQuality] = useState(true)
  const [routingStats, setRoutingStats] = useState({
    totalRequests: 0,
    avgResponseTime: 0,
    totalTokens: 0,
    totalCost: 0,
    modelUsage: {} as Record<string, number>,
    providerUsage: {} as Record<string, number>
  })

  const taskTypes = [
    { value: 'chat', label: '💬 General Chat' },
    { value: 'coding', label: '💻 Code Generation' },
    { value: 'creative', label: '✨ Creative Writing' },
    { value: 'analytical', label: '📊 Analysis & Research' },
    { value: 'summarization', label: '📝 Summarization' },
    { value: 'reasoning', label: '🧠 Complex Reasoning' }
  ]

  const availableModels = [
    { value: '', label: '🧠 Auto-route (Recommended)', provider: 'auto', available: true },
    
    // OpenAI Models (Current)
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo (Fast & Cheap)', provider: 'openai', available: true },
    { value: 'gpt-4', label: 'GPT-4 (Maximum Quality)', provider: 'openai', available: true },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo (Balanced)', provider: 'openai', available: true },
    
    // OpenAI Models (Future - GPT-5 Placeholder)
    { value: 'gpt-5-mini', label: 'GPT-5 Mini (Coming Soon)', provider: 'openai', available: false },
    { value: 'gpt-5', label: 'GPT-5 (Future Release)', provider: 'openai', available: false },
    
    // Anthropic Models  
    { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku (Ultra Fast)', provider: 'anthropic', available: true },
    { value: 'claude-3-sonnet-20240229', label: 'Claude 3 Sonnet (Balanced)', provider: 'anthropic', available: true },
    { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus (Premium)', provider: 'anthropic', available: true },
    
    // Google Gemini Models
    { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash (Lightning Fast)', provider: 'google', available: true },
    { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro (High Performance)', provider: 'google', available: true },
    { value: 'gemini-1.0-pro', label: 'Gemini 1.0 Pro (Reliable)', provider: 'google', available: true }
  ]

  const calculateCost = (usage: any, model: string): number => {
    const costs: Record<string, number> = {
      // OpenAI pricing (per 1K tokens)
      'gpt-3.5-turbo': 0.002,
      'gpt-4': 0.030,
      'gpt-4-turbo': 0.010,
      // Future GPT-5 (estimated pricing)
      'gpt-5-mini': 0.005,
      'gpt-5': 0.040,
      // Anthropic pricing
      'claude-3-haiku-20240307': 0.0025,
      'claude-3-sonnet-20240229': 0.008,
      'claude-3-opus-20240229': 0.015,
      // Google Gemini pricing
      'gemini-1.5-flash': 0.001,
      'gemini-1.5-pro': 0.007,
      'gemini-1.0-pro': 0.005
    }
    const costPerToken = costs[model] || 0.002
    return (usage.total_tokens * costPerToken) / 1000
  }

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Check if selected model is available
    const selectedModel = availableModels.find(m => m.value === forceModel)
    if (selectedModel && !selectedModel.available) {
      alert(`${selectedModel.label} is not yet available. Please select a different model or use auto-routing.`)
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const requestBody = {
        messages: [
          ...messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp.toISOString()
          })),
          {
            role: 'user',
            content: input,
            timestamp: new Date().toISOString()
          }
        ],
        task_type: taskType,
        temperature: 0.7,
        max_tokens: 1000,
        prefer_speed: preferSpeed,
        prefer_quality: preferQuality,
        ...(forceModel && { model: forceModel })
      }

      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.message?.content || 'Sorry, I encountered an error.',
        role: 'assistant',
        timestamp: new Date(),
        routing_info: {
          model: data.model,
          provider: data.provider,
          routing_decision: data.routing_decision,
          response_time_ms: data.response_time_ms,
          usage: data.usage
        }
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Update routing stats
      const cost = calculateCost(data.usage, data.model)
      setRoutingStats(prev => ({
        totalRequests: prev.totalRequests + 1,
        avgResponseTime: Math.round(
          (prev.avgResponseTime * prev.totalRequests + data.response_time_ms) / 
          (prev.totalRequests + 1)
        ),
        totalTokens: prev.totalTokens + data.usage.total_tokens,
        totalCost: prev.totalCost + cost,
        modelUsage: {
          ...prev.modelUsage,
          [data.model]: (prev.modelUsage[data.model] || 0) + 1
        },
        providerUsage: {
          ...prev.providerUsage,
          [data.provider]: (prev.providerUsage[data.provider] || 0) + 1
        }
      }))
      
    } catch (error) {
      console.error('Error sending message:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const getModelDisplayName = (model: string): string => {
    const names: Record<string, string> = {
      // OpenAI
      'gpt-3.5-turbo': 'GPT-3.5 Turbo',
      'gpt-4': 'GPT-4',
      'gpt-4-turbo': 'GPT-4 Turbo',
      'gpt-5-mini': 'GPT-5 Mini',
      'gpt-5': 'GPT-5',
      // Anthropic
      'claude-3-haiku-20240307': 'Claude 3 Haiku',
      'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
      'claude-3-opus-20240229': 'Claude 3 Opus',
      // Google
      'gemini-1.5-flash': 'Gemini 1.5 Flash',
      'gemini-1.5-pro': 'Gemini 1.5 Pro',
      'gemini-1.0-pro': 'Gemini 1.0 Pro'
    }
    return names[model] || model
  }

  const getProviderIcon = (provider: string): string => {
    const icons: Record<string, string> = {
      'openai': '🤖',
      'anthropic': '🎭',
      'google': '💎',
      'simulated': '🧪'
    }
    return icons[provider] || '❓'
  }

  const getProviderColor = (provider: string): string => {
    const colors: Record<string, string> = {
      'openai': 'bg-green-100 text-green-800',
      'anthropic': 'bg-purple-100 text-purple-800', 
      'google': 'bg-blue-100 text-blue-800',
      'simulated': 'bg-gray-100 text-gray-800'
    }
    return colors[provider] || 'bg-gray-100 text-gray-800'
  }

  const clearChat = () => {
    setMessages([])
    setRoutingStats({
      totalRequests: 0,
      avgResponseTime: 0,
      totalTokens: 0,
      totalCost: 0,
      modelUsage: {},
      providerUsage: {}
    })
  }

  return (
    <div className={`flex h-full max-w-7xl mx-auto gap-4 ${className}`}>
      
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="bg-white rounded-lg shadow-sm border p-4 mb-4">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">🧠 Multi-Provider LLM Router</h1>
          <p className="text-gray-600">Intelligent routing across OpenAI, Anthropic, and Google models based on task complexity and preferences.</p>
          <div className="flex gap-2 mt-3">
            <Button onClick={clearChat} variant="outline" size="sm">
              Clear Chat
            </Button>
            <span className="inline-flex items-center px-2 py-1 bg-green-50 border border-green-200 rounded-full text-xs text-green-800">
              <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
              🤖 OpenAI • 🎭 Anthropic • 💎 Google
            </span>
          </div>
        </div>

        {/* Messages Container */}
        <div className="flex-1 bg-white rounded-lg shadow-sm border overflow-hidden flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <div className="text-6xl mb-4">🤖</div>
                <h3 className="text-xl font-semibold mb-2">Test Multi-Provider LLM Routing</h3>
                <p className="mb-4">Try different types of requests to see intelligent routing in action:</p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto text-sm">
                  <div className="bg-blue-50 p-3 rounded cursor-pointer hover:bg-blue-100" 
                       onClick={() => setInput("Write a Python function to calculate fibonacci numbers")}>
                    <strong>💻 Coding:</strong> "Write a Python function to calculate fibonacci numbers"
                  </div>
                  <div className="bg-green-50 p-3 rounded cursor-pointer hover:bg-green-100"
                       onClick={() => setInput("Write a haiku about artificial intelligence")}>
                    <strong>✨ Creative:</strong> "Write a haiku about artificial intelligence"
                  </div>
                  <div className="bg-purple-50 p-3 rounded cursor-pointer hover:bg-purple-100"
                       onClick={() => setInput("Analyze the impact of quantum computing on cybersecurity")}>
                    <strong>📊 Analysis:</strong> "Analyze the impact of quantum computing on cybersecurity"
                  </div>
                  <div className="bg-orange-50 p-3 rounded cursor-pointer hover:bg-orange-100"
                       onClick={() => setInput("Explain the theory of relativity in simple terms")}>
                    <strong>🧠 Reasoning:</strong> "Explain the theory of relativity in simple terms"
                  </div>
                </div>
              </div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className="space-y-2">
                  <div
                    className={`flex ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-2xl px-4 py-3 rounded-lg ${
                        message.role === 'user'
                          ? 'bg-blue-500 text-white'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{message.content}</p>
                      <span className="text-xs opacity-70 mt-2 block">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  
                  {/* Routing Information */}
                  {message.routing_info && (
                    <div className="ml-auto max-w-2xl">
                      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 text-sm">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-indigo-800">🧠 Routing Decision</span>
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getProviderColor(message.routing_info.provider)}`}>
                              {getProviderIcon(message.routing_info.provider)} {message.routing_info.provider}
                            </span>
                          </div>
                          <span className="text-indigo-600">{message.routing_info.response_time_ms}ms</span>
                        </div>
                        <div className="grid grid-cols-2 gap-3 text-xs">
                          <div>
                            <span className="text-gray-600">Model:</span>
                            <span className="ml-1 font-medium">{getModelDisplayName(message.routing_info.model)}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Provider:</span>
                            <span className="ml-1 font-medium capitalize">{message.routing_info.provider}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Tokens:</span>
                            <span className="ml-1 font-medium">{message.routing_info.usage.total_tokens}</span>
                          </div>
                          <div>
                            <span className="text-gray-600">Cost:</span>
                            <span className="ml-1 font-medium">${calculateCost(message.routing_info.usage, message.routing_info.model).toFixed(4)}</span>
                          </div>
                        </div>
                        <div className="mt-2 text-xs text-indigo-700">
                          <strong>Why:</strong> {message.routing_info.routing_decision}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 max-w-2xl px-4 py-3 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                    <p>🧠 Analyzing request and selecting optimal model across providers...</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Form */}
          <form onSubmit={handleSendMessage} className="p-4 border-t bg-gray-50">
            <div className="flex flex-wrap gap-2 mb-3">
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {taskTypes.map(type => (
                  <option key={type.value} value={type.value}>{type.label}</option>
                ))}
              </select>
              
              <select
                value={forceModel}
                onChange={(e) => setForceModel(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 min-w-48"
              >
                {availableModels.map(model => (
                  <option 
                    key={model.value} 
                    value={model.value}
                    disabled={!model.available}
                    className={!model.available ? 'text-gray-400 italic' : ''}
                  >
                    {model.label}
                  </option>
                ))}
              </select>
              
              <div className="flex items-center space-x-3 text-sm">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={preferSpeed}
                    onChange={(e) => setPreferSpeed(e.target.checked)}
                    className="mr-1"
                  />
                  ⚡ Speed
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={preferQuality}
                    onChange={(e) => setPreferQuality(e.target.checked)}
                    className="mr-1"
                  />
                  💎 Quality
                </label>
              </div>
            </div>
            
            <div className="flex space-x-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter your request to test multi-provider LLM routing..."
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isLoading}
              />
              <Button type="submit" disabled={isLoading || !input.trim()}>
                Route & Send
              </Button>
            </div>
          </form>
        </div>
      </div>

      {/* Analytics Sidebar */}
      <div className="w-80 space-y-4">
        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h3 className="font-semibold text-gray-900 mb-3">📊 Session Analytics</h3>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Total Requests:</span>
              <span className="font-medium">{routingStats.totalRequests}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Avg Response Time:</span>
              <span className="font-medium">{routingStats.avgResponseTime}ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Tokens:</span>
              <span className="font-medium">{routingStats.totalTokens.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Cost:</span>
              <span className="font-medium">${routingStats.totalCost.toFixed(4)}</span>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h3 className="font-semibold text-gray-900 mb-3">🏢 Provider Usage</h3>
          <div className="space-y-2 text-sm">
            {Object.entries(routingStats.providerUsage).map(([provider, count]) => (
              <div key={provider} className="flex justify-between items-center">
                <span className="text-gray-600 flex items-center">
                  <span className="mr-1">{getProviderIcon(provider)}</span>
                  {provider}:
                </span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full" 
                      style={{width: `${(count / routingStats.totalRequests) * 100}%`}}
                    ></div>
                  </div>
                  <span className="font-medium w-6 text-right">{count}</span>
                </div>
              </div>
            ))}
            {routingStats.totalRequests === 0 && (
              <p className="text-gray-500 text-center py-4">Send a message to see provider stats</p>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h3 className="font-semibold text-gray-900 mb-3">🤖 Model Usage</h3>
          <div className="space-y-2 text-sm max-h-48 overflow-y-auto">
            {Object.entries(routingStats.modelUsage).map(([model, count]) => (
              <div key={model} className="flex justify-between items-center">
                <span className="text-gray-600 text-xs">{getModelDisplayName(model)}:</span>
                <div className="flex items-center space-x-2">
                  <div className="w-12 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full" 
                      style={{width: `${(count / routingStats.totalRequests) * 100}%`}}
                    ></div>
                  </div>
                  <span className="font-medium w-6 text-right">{count}</span>
                </div>
              </div>
            ))}
            {routingStats.totalRequests === 0 && (
              <p className="text-gray-500 text-center py-4">Send a message to see model stats</p>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h3 className="font-semibold text-gray-900 mb-3">💡 Multi-Provider Benefits</h3>
          <div className="text-xs text-gray-600 space-y-2">
            <p><strong>🤖 OpenAI:</strong> Strong coding, reasoning</p>
            <p><strong>🎭 Anthropic:</strong> Safety, analysis, writing</p>
            <p><strong>💎 Google:</strong> Speed, multimodal, cost-effective</p>
            <p><strong>🔄 Smart routing</strong> maximizes strengths while minimizing costs</p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border p-4">
          <h3 className="font-semibold text-gray-900 mb-2">🚀 Future Ready</h3>
          <div className="text-xs text-gray-600 space-y-1">
            <p>• GPT-5 support (when available)</p>
            <p>• Claude 4 integration ready</p>
            <p>• Gemini 2.0 preparation</p>
            <p>• New providers easy to add</p>
          </div>
        </div>
      </div>
    </div>
  )
}