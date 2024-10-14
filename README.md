# 🚀 Run
Instalar los paquetes necesarios
```py
pip install -r requirements.txt
```
or

```py
python -m pip install -r requirements.txt
```

## Start
Ejecutar el archivo principal
```py
python app.py
```

## Fedora OS:
```bash
sestatus
sudo chcon -Rt svirt_sandbox_file_t ./app
```

### 3. Temporarily Set SELinux to Permissive Mode
If you want to check if SELinux is the issue, you can temporarily set it to permissive mode (this is not recommended for production environments):
```bash
sudo setenforce 0
```
After running this command, try to start your Docker container again. If it works, then SELinux was the problem. Remember to set it back to enforcing mode afterwards:
```bash
sudo setenforce 1
```