import { Button } from '@chakra-ui/react';
import { Toaster } from './Toaster';
import { showToast } from '../utils';

function ToasterUseExample() {
  return (
    <>
      <Button
        variant="surface"
        size="lg"
        onClick={() => {
          showToast({
            title: 'Toast Title',
            description: 'Toast Description',
          });
        }}
      >
        彈出 Toast
      </Button>

      <Toaster />
    </>
  );
}

export default ToasterUseExample;
