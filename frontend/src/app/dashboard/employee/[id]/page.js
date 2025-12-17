// "use client";

// import { useEffect, useState } from "react";
// import { useParams, useRouter } from "next/navigation";
// import { userAPI } from "@/api/user/userAPI";
// import { User, Phone, Mail, BadgeCheck, UserCheck, ArrowLeft } from "lucide-react";

// export default function EmployeePage() {
//   const { id } = useParams();
//   const router = useRouter();
//   const [employee, setEmployee] = useState(null);

//   useEffect(() => {
//     const fetchDataEmployee = async () => {
//       try {
//         const res = await userAPI.getUser(id);
//         setEmployee(res.data);
//       } catch (err) {
//         console.error("Lỗi khi lấy thông tin nhân viên:", err);
//       }
//     };
//     fetchDataEmployee();
//   }, [id]);

//   if (!employee)
//     return (
//       <div className="flex h-64 items-center justify-center text-gray-500">
//         Không tìm thấy nhân viên
//       </div>
//     );

//   return (
//     <div className="mx-auto max-w-md space-y-4">
//       <div className="rounded-xl bg-white p-6 shadow-md">
//         <div className="mb-6 flex items-center space-x-4">
//           {employee.avaUrl ? (
//             <img
//               src={employee.avaUrl}
//               alt={employee.username}
//               className="h-16 w-16 rounded-full object-cover"
//             />
//           ) : (
//             <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gray-200 text-gray-500">
//               <User className="text-green-300" size={32} />
//             </div>
//           )}
//           <div>
//             <h1 className="text-xl font-bold">{employee.username}</h1>
//             <p className="text-sm text-gray-500">ID: {employee.id}</p>
//           </div>
//         </div>

//         <div className="space-y-3 text-gray-700">
//           {employee.phoneNumber && (
//             <div className="flex items-center gap-2">
//               <Phone size={16} className="text-gray-400" />
//               <span>{employee.phoneNumber}</span>
//             </div>
//           )}
//           {employee.email && (
//             <div className="flex items-center gap-2">
//               <Mail size={16} className="text-gray-400" />
//               <span>{employee.email}</span>
//             </div>
//           )}
//           {employee.role?.name && (
//             <div className="flex items-center gap-2">
//               <BadgeCheck size={16} className="text-gray-400" />
//               <span>Role: {employee.role.name}</span>
//             </div>
//           )}
//           {employee.status && (
//             <div className="flex items-center gap-2">
//               <UserCheck size={16} className="text-gray-400" />
//               <span>Status: {employee.status}</span>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { userAPI } from "@/api/user/userAPI";
import { User, Phone, Mail, BadgeCheck, UserCheck, ArrowLeft } from "lucide-react";

export default function EmployeePage() {
  const { id } = useParams();
  const router = useRouter();
  const [employee, setEmployee] = useState(null);

  useEffect(() => {
    const fetchDataEmployee = async () => {
      try {
        const res = await userAPI.getUser(id);
        setEmployee(res.data);
      } catch (err) {
        console.error("Lỗi khi lấy thông tin nhân viên:", err);
      }
    };
    fetchDataEmployee();
  }, [id]);

  if (!employee)
    return (
      <div className="flex h-64 items-center justify-center text-gray-500">
        Không tìm thấy nhân viên
      </div>
    );

  return (
    <div className="mx-auto max-w-md space-y-4 p-4 bg-green-50">
      <div className="rounded-2xl p-6 shadow-lg bg-white">
        {/* Avatar và tên */}
        <div className="mb-6 flex items-center space-x-4 ">
          {employee.avaUrl ? (
            <img
              src={employee.avaUrl}
              alt={employee.username}
              className="h-20 w-20 rounded-full border-2 border-green-400 object-cover"
            />
          ) : (
            <div className="flex h-20 w-20 items-center justify-center rounded-full border-2 border-green-400 text-green-500">
              <User size={36} />
            </div>
          )}
          <div>
            <h1 className="text-2xl font-bold text-gray-800">{employee.username}</h1>
            <p className="text-sm text-gray-500">ID: {employee.id}</p>
          </div>
        </div>

        {/* Thông tin chi tiết */}
        <div className="space-y-3">
          {employee.phoneNumber && (
            <div className="flex items-center gap-3 rounded-lg p-2">
              <Phone size={18} className="text-green-500" />
              <span className="text-gray-700">{employee.phoneNumber}</span>
            </div>
          )}
          {employee.email && (
            <div className="flex items-center gap-3 rounded-lg p-2">
              <Mail size={18} className="text-green-500" />
              <span className="text-gray-700">{employee.email}</span>
            </div>
          )}
          {employee.role?.name && (
            <div className="flex items-center gap-3 rounded-lg  p-2">
              <BadgeCheck size={18} className="text-green-500" />
              <span className="font-medium text-gray-700">Role: {employee.role.name}</span>
            </div>
          )}
          {employee.status && (
            <div className="flex items-center gap-3 rounded-lg  p-2">
              <UserCheck size={18} className="text-green-500" />
              <span className="font-medium text-gray-700">Status: {employee.status}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
