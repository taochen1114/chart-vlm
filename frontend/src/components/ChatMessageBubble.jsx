import { memo } from 'react';
import { Box, Flex } from '@chakra-ui/react';
import { CHAT_BG_CLASSNAMES } from '../config/colorPalette';

const ChatMessageBubble = memo(({ sender, text }) => {
  const isUser = sender === 'user';
  const borr = '2xl';

  return (
    <Flex justify={isUser ? 'flex-end' : 'flex-start'}>
      <Box
        maxWidth="80%"
        px={3}
        py={3}
        borderRadius={borr}
        bg={isUser ? CHAT_BG_CLASSNAMES.user : CHAT_BG_CLASSNAMES.bot}
        borderBottomRightRadius={isUser ? 0 : borr}
        borderBottomLeftRadius={isUser ? borr : 0}
        color={isUser ? '#fff' : ''}
      >
        {text}
      </Box>
    </Flex>
  );
});

export default ChatMessageBubble;
