import { useState } from 'react';
import {
  Button,
  Box,
  FileUpload,
  Input,
  IconButton,
  Flex,
  Separator,
} from '@chakra-ui/react';
import { HiUpload } from 'react-icons/hi';
import { IoIosSend } from 'react-icons/io';
import ChatWindow from './ChatWindow';
import Toaster from './Toaster';
import { showToast } from '../utils';
import { mockChatHistory } from '../mockData/mockChatHistory';

import { pdfjs, Document, Page } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// 方式 1： 透過 local 端資源載入
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

const backendUrl = import.meta.env.VITE_BACKEND_URL;

export default function AskYourFile() {
  const percentSize = ['3/5', '2/5'];

  const [numPages, setNumPages] = useState();
  const [pageNumber, setPageNumber] = useState(1);

  const onDocumentLoadSuccess = (pdf) => {
    // console.log('pdf', pdf);
    setNumPages(pdf.numPages);
  };

  const handlePdfPageChange = (action) => {
    switch (action) {
      case 'prev': {
        setPageNumber((prev) => (prev > 1 ? prev - 1 : prev));
        break;
      }
      case 'next': {
        setPageNumber((prev) => (prev < numPages ? prev + 1 : prev));
        break;
      }
      default: {
        throw new Error('請傳入正確的參數');
      }
    }
  };

  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState(mockChatHistory);
  const clearInput = () => setQuestion('');

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleQuestionChange = (e) => setQuestion(e.target.value);

  const fetchBotResponse = async (text) => {
    const trimmedText = text.trim();

    if (!file) {
      showToast({
        title: '請先選擇 PDF 檔案',
        type: 'error',
      });
      return;
    }
    if (!trimmedText) {
      showToast({
        title: '請輸入想詢問的問題',
        type: 'error',
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', trimmedText);

    try {
      const response = await fetch(`${backendUrl}/analyze-pdf`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP 錯誤：${response.status}`);
      }

      const data = await response.json();
      console.log(data);

      return data;
    } catch (err) {
      showToast({
        title: 'Error',
        description: err.message,
        type: 'error',
      });
    }
  };

  /** 打字機效果 */
  const animateBotTyping = async (fullText) => {
    const segmenter = new Intl.Segmenter('en', { granularity: 'grapheme' });
    const segments = Array.from(segmenter.segment(fullText));
    let displayedText = '';
    const batchSize = 1; // 每 n 個字符更新一次

    setChatHistory((prev) => [
      ...prev,
      {
        sender: 'bot',
        text: '',
        complete: false, // 這條 bot 訊息還沒打完
      },
    ]);

    for (let i = 0; i < segments.length; i++) {
      // 動態設置延遲
      const delay = i === 0 ? 0 : 60;

      displayedText += segments[i].segment;
      if (i % batchSize === 0 || i === segments.length - 1) {
        setChatHistory((prev) => {
          const index = prev.findIndex(
            (msg) => msg.sender === 'bot' && !msg.complete
          );
          if (index === -1) return prev;
          const nextHistory = [...prev];
          nextHistory[index].text = displayedText;
          return nextHistory;
        });
      }
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  };

  const finalizeBotMessage = () => {
    setChatHistory((prev) => {
      const index = prev.findIndex(
        (msg) => msg.sender === 'bot' && !msg.complete
      );
      if (index === -1) {
        return prev;
      }
      const nextHistory = [...prev];
      nextHistory[index].complete = true;
      return nextHistory;
    });
  };

  const sendMessage = async (text) => {
    const trimmedText = text.trim();

    if (!file) {
      showToast({
        title: '請先選擇 PDF 檔案',
        type: 'error',
      });
      return;
    }
    if (!trimmedText) {
      showToast({
        title: '請輸入想詢問的問題',
        type: 'error',
      });
      return;
    }

    clearInput();

    try {
      setChatHistory((prev) => [
        ...prev,
        {
          sender: 'user',
          text: trimmedText,
        },
      ]);

      const response = await fetchBotResponse(trimmedText);
      if (response.answer_page && response.answer_page >= 0) {
        setPageNumber(response.answer_page);
      }
      await animateBotTyping(response.answer);
      finalizeBotMessage();
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      <Flex maxH="100%" gap={8} paddingX={6}>
        {/* <Flex maxH="100%" gap={8} divideColor="blue.200"> */}
        {/* PDF 上傳、PDF 展示 */}
        <Flex
          width={percentSize[0]}
          justifyContent="center"
          alignItems="center"
          flexDirection="column"
        >
          <FileUpload.Root>
            <FileUpload.HiddenInput
              type="file"
              accept="application/pdf"
              onChange={(event) => handleFileChange(event)}
            />
            <Flex justify="center" align="center" width="100%">
              {!file && (
                <FileUpload.Trigger asChild marginBottom={5}>
                  <Button variant="outline" size="lg">
                    <HiUpload /> Upload PDF
                  </Button>
                </FileUpload.Trigger>
              )}
            </Flex>
            <FileUpload.List showSize clearable />
          </FileUpload.Root>

          {/* PDF 上一頁、下一頁 */}
          {/* TODO: 可考慮改用 Chakra 的 pagination */}
          {file && numPages && (
            <Flex justify="center" align="center" gap={3} marginTop={8}>
              <Button
                onClick={() => handlePdfPageChange('prev')}
                colorPalette="blue"
              >
                上一頁
              </Button>
              <p>
                Page {pageNumber} of {numPages}
              </p>
              <Button
                onClick={() => handlePdfPageChange('next')}
                colorPalette="blue"
              >
                下一頁
              </Button>
            </Flex>
          )}

          <Flex direction="column" justify="center" marginTop={3}>
            <Flex
              justify="center"
              style={{ overflowY: 'auto' }}
              maxHeight="68dvh"
            >
              <Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
                <Page pageNumber={pageNumber} />
              </Document>
            </Flex>
          </Flex>
        </Flex>

        <Separator size="md" orientation="vertical" />

        {/* 對話框 */}
        <Box width={percentSize[1]}>
          <ChatWindow chatHistory={chatHistory} />

          <Flex gap="4" justify="center" width="100%" marginTop={8}>
            <Input
              type="text"
              placeholder="請輸入關於 PDF 的問題"
              value={question}
              onChange={handleQuestionChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
                  sendMessage(question.trim());
                }
              }}
            />
            <IconButton
              size="lg"
              colorPalette="blue"
              onClick={() => sendMessage(question.trim())}
            >
              <IoIosSend />
            </IconButton>
          </Flex>
        </Box>
      </Flex>

      <Toaster />
    </>
  );
}
