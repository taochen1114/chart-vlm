import { useState } from 'react';
import { Button, FileUpload, Input } from '@chakra-ui/react';
import { HiUpload } from 'react-icons/hi';
import Toaster from './Toaster';
import { showToast } from '../utils';

const backendUrl = 'http://127.0.0.1:8000';

function PdfUploader() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');

  const handleFileChange = (e) => {
    // console.log('開始處理上傳');
    // console.log(e.target.files);
    setFile(e.target.files[0]);
  };

  const handleQuestionChange = (e) => {
    setQuestion(e.target.value);
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
        title: '請輸入想詢問的問題',
        type: 'error',
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question.trim());

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
    } catch (err) {
      showToast({
        title: 'Error',
        description: err.message,
        type: 'error',
      });
    }
  };

  return (
    <>
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

      <Button size="lg" colorPalette="blue" onClick={handleSubmit}>
        把 PDF 送到後端
      </Button>

      <Toaster />
    </>
  );
}

export default PdfUploader;
