"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

// Data giả toàn bộ ca trực
const fakeSchedule = [
  {
    date: "2025-11-11",
    shift: "08:00-17:30",
    employee: {
      id: 1,
      name: "Nguyen Van A",
      color: "bg-red-200",
      phone: "0123456789",
      note: "Ca sáng",
    },
  },
  {
    date: "2025-11-11",
    shift: "17:30-24:00",
    employee: {
      id: 2,
      name: "Tran Thi B",
      color: "bg-blue-200",
      phone: "0987654321",
      note: "Ca chiều",
    },
  },
  // Có thể thêm nhiều ca khác ở đây
];

export default function EmployeePage() {
  const { id } = useParams();
  const [employee, setEmployee] = useState(null);

  useEffect(() => {
    if (id) {
      // Tìm nhân viên theo id trong fakeSchedule
      const emp = fakeSchedule.find((s) => s.employee?.id.toString() === id)?.employee || null;
      setEmployee(emp);
    }
  }, [id]);

  if (!employee) return <div>Không tìm thấy nhân viên</div>;

  return (
    <div className="rounded-xl bg-white p-6 shadow">
      <h1 className="mb-4 text-2xl font-bold">{employee.name}</h1>
      <p>ID: {employee.id}</p>
      {employee.phone && <p>📞 {employee.phone}</p>}
      {employee.note && <p>📝 {employee.note}</p>}
    </div>
  );
}
