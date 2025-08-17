import { Button } from "@/components/ui/button"
import Link from "next/link"
import { MessageSquare, BarChart3, Settings, Zap, DollarSign, Star } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <MessageSquare className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Intelligent LLM Router</h1>
                <p className="text-sm text-gray-500">Multi-provider optimization</p>
              </div>
            </div>
            
            <nav className="flex items-center space-x-4">
              <Link href="/chat">
                <Button variant="default">Try Router</Button>
              </Link>
              <Link href="/dashboard">
                <Button variant="outline">Dashboard</Button>
              </Link>
            </nav>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            🧠 Intelligent LLM Router
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Automatically route your prompts to the optimal language model (GPT-4, GPT-3.5, Claude) 
            based on complexity, cost, and quality preferences. Save up to <strong>60% on costs</strong> while maintaining 
            excellent response quality.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
            <Link href="/chat">
              <Button size="lg" className="px-8 py-4 text-lg">
                <MessageSquare className="w-5 h-5 mr-2" />
                Start Routing
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button variant="outline" size="lg" className="px-8 py-4 text-lg">
                <BarChart3 className="w-5 h-5 mr-2" />
                View Analytics
              </Button>
            </Link>
          </div>
          
          <div className="inline-flex items-center px-4 py-2 bg-green-50 border border-green-200 rounded-full text-sm text-green-800">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
            OpenAI • Anthropic • Multi-provider routing
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white p-8 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="text-xl font-semibold mb-3">⚡ Smart Classification</h3>
            <p className="text-gray-600 mb-4">
              AI-powered prompt analysis automatically detects task type, complexity, 
              code requests, creative needs, and analytical requirements.
            </p>
            <ul className="text-sm text-gray-500 space-y-1">
              <li>• Complexity scoring (0-100%)</li>
              <li>• Task type detection</li>
              <li>• Context analysis</li>
            </ul>
          </div>
          
          <div className="bg-white p-8 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
              <DollarSign className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold mb-3">💰 Cost Optimization</h3>
            <p className="text-gray-600 mb-4">
              Intelligent routing between models saves 40-60% on costs while 
              maintaining response quality by using the right model for each task.
            </p>
            <ul className="text-sm text-gray-500 space-y-1">
              <li>• GPT-3.5: $0.002/1K tokens</li>
              <li>• GPT-4: $0.030/1K tokens</li>
              <li>• Claude: $0.008/1K tokens</li>
            </ul>
          </div>
          
          <div className="bg-white p-8 rounded-xl shadow-md border border-gray-100 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
              <Star className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-xl font-semibold mb-3">⚡ Performance Tuning</h3>
            <p className="text-gray-600 mb-4">
              Choose your priority (Speed, Cost, Quality) and let our engine 
              optimize routing decisions automatically across providers.
            </p>
            <ul className="text-sm text-gray-500 space-y-1">
              <li>• Speed: 400ms average</li>
              <li>• Cost: 60% savings</li>
              <li>• Quality: 95% retention</li>
            </ul>
          </div>
        </div>

        {/* Model Routing */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-16">
          <h2 className="text-3xl font-bold text-center mb-8">Multi-Provider Model Routing</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center p-6 border-2 border-green-200 rounded-xl bg-green-50">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-semibold text-green-800 mb-2">GPT-3.5 Turbo</h3>
              <p className="text-sm text-green-600 mb-3">Fast & cost-effective</p>
              <div className="space-y-1 text-xs text-green-700">
                <div>$0.002/1K tokens</div>
                <div>~500ms latency</div>
                <div>Chat, Simple tasks</div>
              </div>
            </div>
            
            <div className="text-center p-6 border-2 border-blue-200 rounded-xl bg-blue-50">
              <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <MessageSquare className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-semibold text-blue-800 mb-2">GPT-4</h3>
              <p className="text-sm text-blue-600 mb-3">Maximum capability</p>
              <div className="space-y-1 text-xs text-blue-700">
                <div>$0.030/1K tokens</div>
                <div>~1200ms latency</div>
                <div>Complex reasoning</div>
              </div>
            </div>
            
            <div className="text-center p-6 border-2 border-purple-200 rounded-xl bg-purple-50">
              <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Settings className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-semibold text-purple-800 mb-2">Claude Sonnet</h3>
              <p className="text-sm text-purple-600 mb-3">Balanced performance</p>
              <div className="space-y-1 text-xs text-purple-700">
                <div>$0.008/1K tokens</div>
                <div>~800ms latency</div>
                <div>Analysis, Writing</div>
              </div>
            </div>
            
            <div className="text-center p-6 border-2 border-orange-200 rounded-xl bg-orange-50">
              <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Star className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-semibold text-orange-800 mb-2">Claude Haiku</h3>
              <p className="text-sm text-orange-600 mb-3">Speed specialist</p>
              <div className="space-y-1 text-xs text-orange-700">
                <div>$0.005/1K tokens</div>
                <div>~400ms latency</div>
                <div>Quick responses</div>
              </div>
            </div>
          </div>
        </div>

        {/* Demo Examples */}
        <div className="bg-gray-900 text-white rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center mb-8">See It In Action</h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4 text-green-400">💰 Cost-Optimized Examples</h3>
              <div className="space-y-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Simple Q&A → GPT-3.5 Turbo</div>
                  <div className="text-green-300">"What's the capital of France?"</div>
                  <div className="text-xs text-gray-500 mt-2">Cost: $0.0002 • 500ms • 90% savings vs GPT-4</div>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Quick Tasks → Claude Haiku</div>
                  <div className="text-blue-300">"Summarize this article in 3 sentences"</div>
                  <div className="text-xs text-gray-500 mt-2">Cost: $0.005 • 400ms • 83% savings vs GPT-4</div>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-4 text-blue-400">🎯 Quality-Optimized Examples</h3>
              <div className="space-y-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Complex Coding → GPT-4</div>
                  <div className="text-purple-300">"Build a full React app with authentication..."</div>
                  <div className="text-xs text-gray-500 mt-2">Cost: $0.030 • 1200ms • Maximum quality</div>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-sm text-gray-400 mb-1">Creative Writing → Claude Sonnet</div>
                  <div className="text-orange-300">"Write a compelling marketing story..."</div>
                  <div className="text-xs text-gray-500 mt-2">Cost: $0.008 • 800ms • Creative excellence</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Routing Intelligence */}
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-center mb-8">How The Router Works</h2>
          
          <div className="grid md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-white">1</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Analyze Request</h4>
              <p className="text-sm text-gray-600">Detect task type, complexity, and requirements</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-white">2</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Score Models</h4>
              <p className="text-sm text-gray-600">Evaluate each model's strengths for the task</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-white">3</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Route Optimally</h4>
              <p className="text-sm text-gray-600">Select best model based on your preferences</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-orange-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-white">4</span>
              </div>
              <h4 className="font-semibold text-gray-900 mb-2">Track & Learn</h4>
              <p className="text-sm text-gray-600">Monitor performance and improve routing</p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-6">Ready to Optimize Your LLM Usage?</h2>
          <p className="text-xl text-gray-600 mb-8">
            Start saving costs and improving performance with intelligent routing
          </p>
          
          <Link href="/chat">
            <Button size="lg" className="px-12 py-4 text-lg">
              <MessageSquare className="w-5 h-5 mr-2" />
              Try The Router
            </Button>
          </Link>
        </div>
      </section>
    </div>
  )
}