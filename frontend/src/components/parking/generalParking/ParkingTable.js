"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

export function ParkingTable({ parkedCars }) {
  const [page, setPage] = useState(1);
  const itemsPerPage = 10;
  const startIndex = (page - 1) * itemsPerPage;
  const currentData = parkedCars.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(parkedCars.length / itemsPerPage);

  return (
    <div className="w-full">
      <table className="w-full table-fixed border-collapse text-sm">
        <colgroup>
          <col className="w-[20%]" />
          <col className="w-[25%]" />
          <col className="w-[25%]" />
          <col className="w-[20%]" />
        </colgroup>

        <thead className="sticky top-0 z-10 bg-gray-100 dark:bg-neutral-800">
          <tr>
            <th className="p-3 text-left">Biển số</th>
            <th className="p-3 text-left">Thời gian vào</th>
            <th className="p-3 text-left">Thời gian đỗ</th>
            <th className="p-3 text-left">Trạng thái</th>
          </tr>
        </thead>
      </table>

      <div className="max-h-70 overflow-y-auto">
        <table className="w-full table-fixed border-collapse text-sm">
          <colgroup>
            <col className="w-[20%]" />
            <col className="w-[25%]" />
            <col className="w-[25%]" />
            <col className="w-[20%]" />
          </colgroup>

          <tbody>
            {currentData.map((row, index) => (
              <tr key={index} className="border-b hover:bg-gray-50 dark:hover:bg-neutral-900">
                <td className="p-3 text-left">{row.plate}</td>
                <td className="p-3 text-left">{row.timeIn}</td>
                <td className="p-3 text-left">{row.duration}</td>
                <td className="p-3 text-left">{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex justify-end gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page === 1}
        >
          &lt;
        </Button>

        {[...Array(totalPages)].map((_, i) => (
          <Button
            key={i}
            variant={i + 1 === page ? "default" : "outline"}
            size="sm"
            onClick={() => setPage(i + 1)}
          >
            {i + 1}
          </Button>
        ))}

        <Button
          variant="outline"
          size="sm"
          onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          disabled={page === totalPages}
        >
          &gt;
        </Button>
      </div>
    </div>
  );
}
