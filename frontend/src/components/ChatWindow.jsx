import { useEffect, useRef } from 'react';
import { Flex } from '@chakra-ui/react';
import ChatMessageBubble from './ChatMessageBubble';

export default function ChatWindow({ chatHistory, isInConversation }) {
  const scrollRef = useRef(null);

  // 滾動到底部
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  return (
    <Flex overflowY="auto" maxH="80dvh" direction="column" gap={3}>
      {chatHistory.map((chat, index) => (
        <ChatMessageBubble key={index} sender={chat.sender} text={chat.text} />
      ))}
      <div ref={scrollRef} style={{ marginTop: 0 }} />
    </Flex>
  );
}
