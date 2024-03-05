#!/bin/sh
openssl enc -d -a -aes-256-cbc -pbkdf2 -kfile randar_2022_password.txt < randar_2022.txt.enc
