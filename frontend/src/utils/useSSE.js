// "use client";
// import { useEffect, useRef } from "react";

// export function useSSE(url, eventName, callback) {
//   const esRef = useRef(null);

//   useEffect(() => {
//     if (esRef.current) return;

//     const es = new EventSource(url);
//     esRef.current = es;

//     const handler = (e) => {
//       try {
//         callback(JSON.parse(e.data));
//       } catch (err) {
//         console.error("Parse SSE error", err);
//       }
//     };

//     es.addEventListener(eventName, handler);

//     es.onerror = (e) => {
//       console.error("SSE error:", e);
//     };

//     return () => {
//       es.removeEventListener(eventName, handler);
//       es.close();
//       esRef.current = null;
//     };
//   }, [url, eventName, callback]);
// }
"use client";
import { useEffect, useRef } from "react";

export function useSSE(url, eventName, callback, cacheKey) {
  const esRef = useRef(null);

  useEffect(() => {
    if (cacheKey) {
      const cached = localStorage.getItem(cacheKey);
      if (cached) {
        try {
          callback(JSON.parse(cached), true); 
        } catch {}
      }
    }

    if (esRef.current) return;

    const es = new EventSource(url);
    esRef.current = es;

    const handler = (e) => {
      try {
        const data = JSON.parse(e.data);

        // 🔹 Lưu cache
        if (cacheKey) {
          localStorage.setItem(cacheKey, JSON.stringify(data));
        }

        callback(data, false);
      } catch (err) {
        console.error("Parse SSE error", err);
      }
    };

    es.addEventListener(eventName, handler);

    es.onerror = (e) => {
      console.error("SSE error:", e);
    };

    return () => {
      es.removeEventListener(eventName, handler);
      es.close();
      esRef.current = null;
    };
  }, [url, eventName]);
}
