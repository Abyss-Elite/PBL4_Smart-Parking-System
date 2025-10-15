"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import authAPI from "@/api/auth/authApi";
import PATH from "@/routes/PATH";
import { setToken } from "@/utils/tokenStorage";

export default function LoginForm({ onShowReset }) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const handleLogin = async () => {
    try {
      const res = await authAPI.login({ email, password });

      const data = res.data;
      localStorage.setItem("username", data.username);
      localStorage.setItem("role", data.role);
      localStorage.setItem("userId", data.id);
      setToken({ accessToken: data.accessToken });

      if (data.role === "ADMIN") router.push(PATH.DASHBOARD.ADMIN_HOME);
      else router.push("/home");
    } catch (err) {
      setErrorMessage(err.response?.data?.message || "Sai tài khoản hoặc mật khẩu!");
    }
  };

  return (
    <Card className="w-full max-w-sm border-none shadow-lg">
      <CardHeader>
        <CardTitle className="text-center text-2xl font-bold text-gray-800">Đăng nhập</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Input
          placeholder="Email đăng nhập"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          placeholder="Mật khẩu"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        {errorMessage && (
          <Alert variant="destructive">
            <AlertDescription>{errorMessage}</AlertDescription>
          </Alert>
        )}
        <Button
          onClick={handleLogin}
          className="w-full bg-blue-600 font-bold text-white hover:bg-blue-700"
        >
          Đăng nhập
        </Button>
        <div className="space-y-2 text-center">
          <button onClick={onShowReset} className="text-sm text-blue-600 hover:text-green-600">
            Quên mật khẩu?
          </button>
          <p className="text-sm text-gray-600">
            Bạn chưa có tài khoản?{" "}
            <a href={PATH.REGISTER} className="text-blue-600 hover:text-green-600">
              Đăng ký
            </a>
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
