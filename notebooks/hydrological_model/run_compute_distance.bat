@echo off
set "PYTHON_SCRIPT_PATH=C:\Users\avpalves\Downloads\Nova pasta\ML-ModClark-IUH-Model\notebooks\hydrology\compute_distance_grid.py"

:start_loop
echo.
echo --- Iniciando a execucao do script Python ---
python "%PYTHON_SCRIPT_PATH%"

REM Verifica o código de saída (ERRORLEVEL). Se for diferente de 0, significa que houve um erro.
if %ERRORLEVEL% neq 0 (
    echo.
    echo ATENCAO: O script '%PYTHON_SCRIPT_PATH%' falhou com o codigo de saida %ERRORLEVEL%.
    echo Reiniciando em 1 segundo...
    
    REM O comando 'timeout' é o equivalente ao 'sleep' do Linux.
    timeout /t 1 /nobreak > nul
    
    REM Volta para o início do loop.
    goto start_loop
)

echo.
echo --- Script Python executado com sucesso! ---
pause