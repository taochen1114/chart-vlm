import { Box, Heading } from '@chakra-ui/react';
// import PdfReader from './components/PdfReader';
// import PdfUploader from './components/PdfUploader';
// import PdfUploader_Demo from './components/PdfUploaderDemo';
// import ToasterUseExample from './components/ToasterUseExample';
import AskYourFile from './components/AskYourFile';
import './App.css';

function App() {
  return (
    <Box id="app" maxH="100dvh" overflowY="auto">
      <Heading size="4xl" textAlign="center" marginY={5}>
        拷問你的 PDF
      </Heading>
      <AskYourFile />
    </Box>
  );
}

export default App;
