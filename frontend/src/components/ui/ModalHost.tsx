/**
 * @file src/components/ui/ModalHost.tsx
 * @description Central modal renderer that displays app-level alerts and confirmations.
 */

import type { ModalConfig } from "../../types/data";
import { AppButton } from "./AppButton";

interface ModalHostProps {
  modal: ModalConfig | null;
  onClose: () => void;
}

export function ModalHost({ modal, onClose }: ModalHostProps) {
  if (!modal) {
    return null;
  }

  const actions = modal.actions && modal.actions.length > 0
    ? modal.actions
    : [{ label: "Close", onClick: onClose }];

  return (
    <div className="modal-overlay" role="presentation" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <h2 className="modal__title">{modal.title}</h2>
        {modal.message ? <p className="modal__message">{modal.message}</p> : null}
        {modal.content ? <pre className="modal__content">{modal.content}</pre> : null}

        <div className="modal__actions">
          {actions.map((action) => (
            <AppButton
              key={action.label}
              variant={action.kind === "danger" ? "danger" : action.kind === "primary" ? "primary" : "default"}
              onClick={() => {
                action.onClick?.();
                if (!action.onClick) {
                  onClose();
                }
              }}
            >
              {action.label}
            </AppButton>
          ))}
        </div>
      </div>
    </div>
  );
}





