import { useId } from "react";

/** The OneClick mark: a cursor mid-click on a Samsung-blue tile. */
export function Mark({ className = "" }: { className?: string }) {
  const id = useId();
  return (
    <svg className={`ocm ${className}`} viewBox="0 0 40 40" aria-hidden>
      <defs>
        <linearGradient id={`${id}bg`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#4f88ff" />
          <stop offset="1" stopColor="#1428a0" />
        </linearGradient>
        <linearGradient id={`${id}gl`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fff" stopOpacity=".3" />
          <stop offset=".55" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
      </defs>
      <rect width="40" height="40" rx="12" fill={`url(#${id}bg)`} />
      <rect width="40" height="40" rx="12" fill={`url(#${id}gl)`} />
      <rect x=".5" y=".5" width="39" height="39" rx="11.5" fill="none" stroke="#fff" strokeOpacity=".16" />
      <g className="ocm-burst" stroke="#d4ff3a" strokeWidth="2.6" strokeLinecap="round" fill="none">
        <path d="M15.2 8.4V5.6" />
        <path d="M10.4 13.2H7.6" />
        <path d="M11.8 9.8 9.8 7.8" />
      </g>
      <path
        className="ocm-cursor"
        d="M15.2 13.2V29.2L19.5 25.4 22.3 32.1 25.2 30.9 22.4 24.6H27.6Z"
        fill="#fff"
        stroke="#fff"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
    </svg>
  );
}
