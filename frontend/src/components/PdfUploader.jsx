import { useState } from 'react';
import { Button, FileUpload, Input, Stack } from '@chakra-ui/react';
import { HiUpload } from 'react-icons/hi';
import Toaster from './Toaster';
import { toaster } from '../utils';

const backendUrl = 'http://127.0.0.1:8000';

function PdfUploader() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');

  const handleFileChange = (e) => {
    // console.log('開始處理上傳');
    // console.log(e.target.files);
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      showToast({
        title: '請先選擇 PDF 檔案',
        type: 'error',
      });
      return;
    }
    if (!question.trim()) {
      showToast({
        title: '請輸入問題',
        type: 'error',
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);

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

      if (data.answer_page) {
        console.log(`answer_page: ${data.answer_page}`);
      }
      // setResult(data);
      // setError(null);
    } catch (err) {
      showToast({
        title: 'Error',
        description: err.message,
      });
      // setError(err.message);
      // setResult(null);
    }
  };

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
  };

  const showToast = ({ title, description, type = 'info' }) => {
    toaster.create({
      title,
      description,
      type,
      duration: 1500,
      closable: true,
    });
  };

  return (
    <>
      <Stack gap="5" style={{ padding: '20px' }}>
        <FileUpload.Root>
          <FileUpload.HiddenInput
            type="file"
            accept="application/pdf"
            onChange={(event) => handleFileChange(event)}
          />

          <FileUpload.Trigger asChild>
            <Button variant="outline" size="lg">
              <HiUpload /> Upload file
            </Button>
          </FileUpload.Trigger>
          <FileUpload.List showSize clearable />
        </FileUpload.Root>

        <Input
          type="text"
          placeholder="請輸入關於 PDF 的問題"
          value={question}
          onChange={handleQuestionChange}
        />

        <Button
          // variant="outline"
          size="lg"
          onClick={handleSubmit}
          colorPalette="blue"
        >
          把 PDF 送到後端
        </Button>
      </Stack>

      <Toaster />
    </>
  );
}

export default PdfUploader;
