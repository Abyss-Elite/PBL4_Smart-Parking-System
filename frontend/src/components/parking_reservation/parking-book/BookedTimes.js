"use client";

export default function BookedTimes({ booked, bookingMode }) {
  const hasBooked = booked && booked.length > 0;

  return (
    <div className="space-y-2">
      <p className="font-medium">
        {bookingMode === "week" ? "Tuần đã được đặt" : "Tháng đã được đặt"}
      </p>

      {!hasBooked ? (
        <p className="text-sm text-gray-500">Chưa có ai đặt</p>
      ) : (
        booked.map((b, idx) => (
          <div key={idx} className="flex items-center gap-2 text-red-500">
            ● <span>{b}</span>
          </div>
        ))
      )}
    </div>
  );
}
