import { createToaster } from '@chakra-ui/react';

export const toaster = createToaster({
  placement: 'bottom-end',
  pauseOnPageIdle: true,
});

export const showToast = ({ title, description, type = 'info' }) => {
  toaster.create({
    title,
    description,
    type,
    duration: 1500,
    closable: true,
  });
};
