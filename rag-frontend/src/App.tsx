import { useState, useRef, useEffect, type FormEvent } from 'react';
import { sendQuery } from './api';
import type { Message } from './types';
import { ChatMessage } from './components/ChatMessage';

function makeId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const query = input.trim();
    if (!query || isLoading) return;

    const userMessage: Message = { id: makeId(), role: 'user', content: query };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendQuery(query);
      const assistantMessage: Message = {
        id: makeId(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        roundsUsed: response.rounds_used,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Codebase RAG Assistant</h1>
        <p className="muted">Ask questions about the indexed repository.</p>
      </header>

      <div className="chat-window">
        {messages.length === 0 && (
          <p className="empty-state">
            Try asking something like "how does gallery deletion work?"
          </p>
        )}
        {messages.map((m) => (
          <ChatMessage key={m.id} message={m} />
        ))}
        {isLoading && <div className="message message-assistant loading">Thinking…</div>}
        {error && <div className="error-banner">{error}</div>}
        <div ref={bottomRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the codebase…"
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}