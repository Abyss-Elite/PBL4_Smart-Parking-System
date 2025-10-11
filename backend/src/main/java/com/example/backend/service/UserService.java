package com.example.backend.service;
import com.example.backend.model.User;
import com.example.backend.model.Car;
import com.example.backend.model.Role;
import com.example.backend.repository.CarRepository;
import com.example.backend.repository.UserRepository;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;
import java.util.List;
@Service
public class UserService {
    @Autowired
    private CarRepository carRepository;
    private final UserRepository userRepository;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    public UserService(UserRepository userRepository){
        this.userRepository = userRepository;
    }

    public User register(String username, String password, Role role, String email, String phoneNumber, String status, Boolean isDelete){
        if(userRepository.findByUsername(username).isPresent()){
            throw new RuntimeException("Tên đăng nhập đã tồn tại.");
        }
        if(userRepository.findByEmail(email).isPresent()){
            throw new RuntimeException("Email đã tồn tại");
        }
        if(userRepository.findByPhoneNumber(phoneNumber).isPresent()){
            throw new RuntimeException("Số điện thoại đã được đăng ký.");
        }
        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        user.setRole(role);
        user.setEmail(email);
        user.setStatus(status);
        user.setDelete(isDelete);
        user.setPhoneNumber(phoneNumber);

        return userRepository.save(user);
    }

    public boolean login(String email, String password){
        Optional<User> userOpt = userRepository.findByEmail(email);
        if(userOpt.isPresent()){
            User user = userOpt.get();
            return passwordEncoder.matches(password, user.getPassword());
        }
        return false;
    }

    public boolean resetPassword(String username, String email, String newpassword){
        Optional<User> userOpt = userRepository.findByUsername(username);
        if(userOpt.isPresent()){
            User user = userOpt.get();
            if(user.getEmail().equals(email)){
                user.setPassword(passwordEncoder.encode(newpassword));
                userRepository.save(user);
                return true;
            }else{
                return false;
            }
        }else{
            return false;
        }
    }

    public List<User> getAllUsers(){
        return userRepository.findAll();
    }
    public User createUser(User user){
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        return userRepository.save(user);
    }
    public User getUser(Long id){
    return userRepository.findById(id)
            .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng."));
    }

    public User updateUser(Long id, User userDetails) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Không tìm thấy người dùng."));

        if (userDetails.getPassword() != null && !userDetails.getPassword().isBlank()) {
            user.setPassword(passwordEncoder.encode(userDetails.getPassword()));
        }

        if (userDetails.getEmail() != null && !userDetails.getEmail().isBlank()) {
            user.setEmail(userDetails.getEmail());
        }

        if (userDetails.getUsername() != null && !userDetails.getUsername().isBlank()) {
            user.setUsername(userDetails.getUsername());
        }

        if (userDetails.getPhoneNumber() != null && !userDetails.getPhoneNumber().isBlank()) {
            user.setPhoneNumber(userDetails.getPhoneNumber());
        }

        if (userDetails.getAvaUrl() != null && !userDetails.getAvaUrl().isBlank()) {
            user.setAvaUrl(userDetails.getAvaUrl());
        }

        if (userDetails.getRole() != null && userDetails.getRole().getId() != null) {
            user.setRole(userDetails.getRole());
        }

        return userRepository.save(user);
    }



    public void deleteUser(Long id){
        userRepository.deleteById(id);
    }

    public boolean addCartoUser(Long userId, Car car){
        return userRepository.findById(userId).map(user -> {
            car.setOwner(user);
            carRepository.save(car);
            return true;
        }).orElse(false);
    }
}
