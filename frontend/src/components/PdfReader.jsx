import { useState } from 'react';
import { pdfjs, Document, Page } from 'react-pdf';
import { Button } from '@chakra-ui/react';

import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// 方式 1： 透過 local 端資源載入
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

// 方式 2：  透過 cdn 載入
// pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

function PdfReader() {
  const [numPages, setNumPages] = useState();
  const [pageNumber, setPageNumber] = useState(1);

  const onDocumentLoadSuccess = (pdf) => {
    // console.log('pdf', pdf);
    setNumPages(pdf.numPages);
  };

  // const file = {
  //   url: 'http://localhost:5555/1Q25_MP_Chinese_Vupload.pdf',
  // };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        flexDirection: 'column',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          border: '2px solid gray',
          borderRadius: '4px',
        }}
      >
        <Document
          file="/1Q25_MP_Chinese_Vupload.pdf"
          // file={file}
          onLoadSuccess={onDocumentLoadSuccess}
        >
          <Page pageNumber={pageNumber} />
        </Document>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          marginTop: '20px',
        }}
      >
        <p>
          Page {pageNumber} of {numPages}
        </p>
        <Button
          onClick={() => {
            setPageNumber(60);
          }}
          variant="surface"
          style={{ marginLeft: '20px' }}
        >
          前往最後一頁 (第 {numPages} 頁)
        </Button>
      </div>
    </div>
  );
}

export default PdfReader;
