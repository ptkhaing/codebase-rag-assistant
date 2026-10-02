import ReactMarkdown from 'react-markdown';
import type { Message } from '../types';

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`message message-${message.role}`}>
      <div className="message-content">
        {isUser ? message.content : <ReactMarkdown>{message.content}</ReactMarkdown>}
      </div>

      {!isUser && message.sources && message.sources.length > 0 && (
        <div className="sources">
          <p className="sources-label">
            Sources {message.roundsUsed === 2 ? '(2 retrieval rounds)' : ''}
          </p>
          <ul>
            {message.sources.map((s, i) => (
              <li key={i} className="source-chip">
                {s.file_path}:{s.start_line}-{s.end_line}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}