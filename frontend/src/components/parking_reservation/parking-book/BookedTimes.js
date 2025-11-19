"use client";

export default function BookedTimes({ booked }) {
  return (
    <div className="space-y-2">
      <p className="font-medium">Khung giờ đã có người đặt</p>
      {booked.length === 0 && <p className="text-sm text-gray-500">Chưa có ai đặt</p>}

      {booked.map((t, idx) => (
        <div key={idx} className="flex items-center gap-2 text-red-500">
          ●{" "}
          <span>
            {t.start} → {t.end}
          </span>
        </div>
      ))}
    </div>
  );
}
