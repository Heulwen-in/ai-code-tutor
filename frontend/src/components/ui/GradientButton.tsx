import type { ButtonHTMLAttributes, ReactNode } from "react";

type GradientButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
};

export function GradientButton({ children, className = "", ...rest }: GradientButtonProps) {
  return (
    <button className={`btn btn-gradient ${className}`.trim()} {...rest}>
      {children}
    </button>
  );
}
