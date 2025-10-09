import smtplib

try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login('motivatedmonk2893@gmail.com', 'zkgbvmaotbyaxwhz')
    print("Login successful!")
    server.quit()
except Exception as e:
    print("Login failed with error:", e)