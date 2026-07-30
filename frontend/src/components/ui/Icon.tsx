import type { CSSProperties } from "react";

type IconProps = {
  name: string;
  size?: number;
  fill?: boolean;
  className?: string;
  style?: CSSProperties;
};

// Renders a Material Symbols Rounded glyph. `name` is the icon's ligature id
// (e.g. "dashboard"); `size` overrides the font-size, `fill` toggles the solid
// variant. Decorative by default — pass an accessible label on the parent.
export function Icon({ name, size, fill, className, style }: IconProps) {
  return (
    <span
      aria-hidden
      className={className ? `material-symbols-rounded ${className}` : "material-symbols-rounded"}
      style={{
        fontSize: size,
        fontVariationSettings: fill ? '"FILL" 1' : undefined,
        ...style,
      }}
    >
      {name}
    </span>
  );
}
