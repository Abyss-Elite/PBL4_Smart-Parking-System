// "use client";

// import { useEffect, useState } from "react";
// import Image from "next/image";
// import { userAPI } from "@/api/user/userAPI";

// export default function ProfilePage() {
//   const [form, setForm] = useState();
//   useEffect(() => {
//     const fetchInformationData = async () => {
//       const res = await userAPI.information();
//       setForm(res.data);
//     };
//     fetchInformationData();
//   }, []);

//   const [initialData] = useState(form);

//   const handleChange = (e) => {
//     setForm({
//       ...form,
//       [e.target.name]: e.target.value,
//     });
//   };

//   const handleSave = () => {
//     console.log("Saving data:", form);
//     alert("Saved!");
//   };

//   const handleReset = () => {
//     setForm(initialData);
//   };

//   return (
//     <div className="flex w-full justify-center py-10">
//       <div className="w-[700px] rounded-2xl border bg-white p-10 shadow-sm">
//         <h2 className="mb-2 text-2xl font-semibold">Profile Settings</h2>
//         <p className="mb-8 text-gray-500">Manage your account settings and profile information.</p>

//         {/* --- AVATAR + URL --- */}
//         <div className="mb-8 flex items-start gap-6">
//           <Image
//             src={form.avatar}
//             width={90}
//             height={90}
//             alt="avatar"
//             className="rounded-full border"
//           />

//           <div className="w-full">
//             <label className="mb-2 block font-medium text-gray-700">Avatar URL</label>
//             <input
//               type="text"
//               name="avatar"
//               value={form.avatar}
//               onChange={handleChange}
//               className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
//             />
//           </div>
//         </div>

//         {/* --- FULL NAME --- */}
//         <div className="mb-6">
//           <label className="mb-2 block font-medium text-gray-700">Full Name</label>
//           <input
//             type="text"
//             name="fullName"
//             value={form.fullName}
//             onChange={handleChange}
//             className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
//           />
//         </div>

//         {/* --- EMAIL ADDRESS --- */}
//         <div className="mb-10">
//           <label className="mb-2 block font-medium text-gray-700">Email Address</label>
//           <input
//             type="email"
//             name="email"
//             value={form.email}
//             onChange={handleChange}
//             className="w-full rounded-lg border px-4 py-2 text-sm focus:ring-2 focus:ring-black"
//           />
//         </div>

//         {/* --- BUTTONS --- */}
//         <div className="flex gap-4">
//           <button
//             onClick={handleSave}
//             className="rounded-lg bg-black px-6 py-2 text-white hover:bg-gray-800"
//           >
//             Save Changes
//           </button>

//           <button onClick={handleReset} className="rounded-lg border px-6 py-2 hover:bg-gray-100">
//             Reset
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// }

"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { userAPI } from "@/api/user/userAPI";

export default function ProfilePage() {
  const [form, setForm] = useState(null);
  const [initialData, setInitialData] = useState(null);
  const [notification, setNotification] = useState(null);
  const list = {
    id: 3,
    username: "linh123",
    phoneNumber: "0123456789",
    email: "linh@example.com",
    avaUrl: null,
    status: "ACTIVE",
    role: {
      id: 3,
      name: "USER",
    },
    delete: false,
  };
  useEffect(() => {
    const fetchInformationData = async () => {
      try {
        // const res = await userAPI.information();
        // setForm(res.data);
        // setInitialData(res.data);
        setForm(list)
        setInitialData(list);
      } catch (err) {
        setNotification({
          type: "error",
          message: "Không thể tải thông tin người dùng.",
        });
      }
    };
    fetchInformationData();
  }, []);

  if (!form) return <div className="p-10 text-center">Đang tải...</div>;

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const handleSave = async () => {
    try {
      await userAPI.updateUser(form);
      setNotification({
        type: "success",
        message: "Cập nhật thông tin thành công!",
      });
    } catch (err) {
      setNotification({
        type: "error",
        message: "Không thể lưu thay đổi.",
      });
    }
  };

  const handleReset = () => {
    setForm(initialData);
    setNotification({
      type: "info",
      message: "Mọi thay đổi đã được phục hồi.",
    });
  };

  const getNotificationClass = (type) => {
    switch (type) {
      case "success":
        return "bg-green-400 text-white";
      case "error":
        return "bg-red-400 text-white";
      case "info":
        return "bg-gray-400 text-white";
      default:
        return "";
    }
  };

  return (
    <div className="flex w-full justify-center py-10">
      <div className="w-[700px] rounded-2xl border bg-white p-10 shadow-sm">
        <h2 className="mb-2 text-2xl font-semibold">Thông tin cá nhân</h2>
        <p className="mb-4 text-gray-500">Quản lý và cập nhật thông tin tài khoản của bạn.</p>

        {/* Hiển thị notification */}
        {notification && (
          <div className={`mb-6 rounded p-3 ${getNotificationClass(notification.type)}`}>
            {notification.message}
          </div>
        )}

        {/* Avatar */}
        <div className="mb-8 flex items-start gap-6">
          <Image
            src={form.avaUrl || "/default-avatar.png"}
            width={90}
            height={90}
            alt="avatar"
            className="rounded-full border object-cover"
          />
          <div className="w-full">
            <label className="mb-2 block font-medium text-gray-700">Avatar URL</label>
            <input
              type="text"
              name="avaUrl"
              value={form.avaUrl || ""}
              onChange={handleChange}
              className="w-full rounded-lg border px-4 py-2 text-base focus:ring-2 focus:ring-black"
            />
          </div>
        </div>

        {/* Username */}
        <div className="mb-6">
          <label className="mb-2 block font-medium text-gray-700">Username</label>
          <input
            type="text"
            name="username"
            value={form.username}
            onChange={handleChange}
            className="w-full rounded-lg border px-4 py-2 text-base focus:ring-2 focus:ring-black"
          />
        </div>

        {/* Email */}
        <div className="mb-6">
          <label className="mb-2 block font-medium text-gray-700">Email</label>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            className="w-full rounded-lg border px-4 py-2 text-base focus:ring-2 focus:ring-black"
          />
        </div>

        {/* Phone */}
        <div className="mb-6">
          <label className="mb-2 block font-medium text-gray-700">Số điện thoại</label>
          <input
            type="text"
            name="phoneNumber"
            value={form.phoneNumber}
            onChange={handleChange}
            className="w-full rounded-lg border px-4 py-2 text-base focus:ring-2 focus:ring-black"
          />
        </div>

        {/* Role */}
        <div className="mb-10">
          <label className="mb-2 block font-medium text-gray-700">Vai trò</label>
          <input
            type="text"
            value={form.role?.name}
            disabled
            className="w-full rounded-lg border bg-gray-100 px-4 py-2 text-base"
          />
        </div>

        {/* Buttons */}
        <div className="flex gap-4">
          <button
            onClick={handleSave}
            className="rounded-lg bg-black px-6 py-2 text-white hover:bg-gray-800 cursor-pointer"
          >
            Lưu thay đổi
          </button>
          <button onClick={handleReset} className="rounded-lg border px-6 py-2 hover:bg-gray-100 cursor-pointer">
            Hoàn tác
          </button>
        </div>
      </div>
    </div>
  );
}
