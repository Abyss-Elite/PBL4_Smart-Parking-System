"use client";

export default function SelectedCars({ cars }) {
  return (
    <div className="mt-4 space-y-3">
      <p className="font-medium">Xe bạn đã chọn</p>

      {cars.map((c, i) => (
        <div key={i} className="rounded-xl bg-white p-3 shadow">
          <p className="font-semibold">
            {c.slot} — {c.plate}
          </p>
          {c.mode === "week" && <p className="text-sm text-gray-500">Tuần: {c.weekStart}</p>}
          {c.mode === "month" && <p className="text-sm text-gray-500">Tháng: {c.month}</p>}
        </div>
      ))}
    </div>
  );
}
