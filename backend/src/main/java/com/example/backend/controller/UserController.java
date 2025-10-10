package com.example.backend.controller;
import com.example.backend.model.User;
import com.example.backend.model.Car;
import com.example.backend.service.UserService;
import com.example.backend.repository.UserRepository;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;
// import jakarta.websocket.server.PathParam;

import java.util.Optional;
import java.util.Map;
import java.util.List;




@RestController
@RequestMapping("/api/user")
@CrossOrigin(origins = "*") //cho phép frontend gọi API
public class UserController {
    private final UserService userService;
    private final UserRepository userRepository;
    
    public UserController(UserService userService, UserRepository userRepository){
        this.userService = userService;
        this.userRepository = userRepository;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody User user) {
        try {
            userService.register(
                    user.getUsername(),
                    user.getPassword(),
                    user.getRole(),
                    user.getEmail()
            );
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Đăng ký thành công"
            ));
        } catch (RuntimeException e) {
            return ResponseEntity.badRequest().body(Map.of(
                "message", e.getMessage()
            ));
        }
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody User user) {
        boolean success = userService.login(user.getUsername(), user.getPassword());
        if (success) {
            Optional<User> u = userRepository.findByUsername(user.getUsername());
            User u2 = u.get();
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Đăng nhập thành công",
                "username", u2.getUsername(),
                "email", u2.getEmail(),
                "role", u2.getRole(),
                "id", u2.getId()
            ));
        } else {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of(
                "status", "error",
                "message", "Tài khoản không tồn tại."
            ));
        }
    }

    @PostMapping("/reset-password")
    public ResponseEntity<?> resetPassword(@Valid @RequestBody User user){
        boolean success = userService.resetPassword(user.getUsername(), user.getEmail(), user.getPassword());
        if(success){
            return ResponseEntity.ok(Map.of(
                "status", "success",
                "message", "Đặt lại mật khẩu thành công!"
            ));
        }else{
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of(
                "status", "error",
                "message", "Tài khoản không tồn tại."
            ));
        }
    }

    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping("/getAllUsers")
    public List<User> getAllUsers(){
        return userService.getAllUsers();
    }

    @PreAuthorize("hasRole('ADMIN')")
    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id){
        return userService.getUser(id);
    }

    @PreAuthorize("hasRole('ADMIN')")
    @PostMapping()
    public User createUser(@RequestBody User user){
        return userService.createUser(user);
    }

    @PreAuthorize("hasRole('ADMIN')")
    @PutMapping("/{id}")
    public User updateUser(@PathVariable Long id, @RequestBody User userDetails) {
        return userService.updateUser(id, userDetails);
    }

    @PreAuthorize("hasRole('ADMIN')")
    @DeleteMapping("/{id}")
    public void deleteUser(@PathVariable Long id) {
        userService.deleteUser(id);
    }

    
    @PostMapping("/{userId}/car")
    public ResponseEntity<Car> addCartoUser(@PathVariable Long userId, @RequestBody Car car){
        boolean success = userService.addCartoUser(userId, car);
        if(success){
            return ResponseEntity.ok().body(car);
        }else{
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).build();
        }
    }

}
