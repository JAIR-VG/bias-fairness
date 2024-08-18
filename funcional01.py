respuesta=0

#Funcion
def suma(x,y):
    return x + y


#Se imprime el resultado al llamar la función
print(suma(3,4))

#El resultado de la función se guarda en respuesta
respuesta = suma(3,4)

#Se imprime el contenido de la variable respuesta
print(respuesta)

#Crea una nueva referencia a suma
respuesta = suma

print('Esto es respuesta', respuesta(3,4))

#Comprobar que respuesta solo hace referencia
print(suma)
print(respuesta)