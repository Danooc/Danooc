# Danooc

IA para call center tipo IVR ("presione 1 para ventas...") con clasificador de intenciones entrenable y servidor compatible con Twilio.

---

## Qué hace

- **IVR inteligente**: menú de opciones por teclado (DTMF) **y** por voz con IA
- **Clasificador de intenciones**: modelo de NLP entrenable que entiende lenguaje natural en español
- **Flujo de llamada configurable**: define menús, submenús, transferencias, buzón de voz — todo en un JSON
- **Servidor Twilio-ready**: webhooks FastAPI que generan TwiML para integrar con Twilio
- **Simulador de terminal**: prueba el sistema completo sin necesidad de teléfono

## Estructura del proyecto

```
danooc/
├── ivr/
│   ├── engine.py      # Motor de flujo de llamada (máquina de estados)
│   └── models.py      # Modelos: MenuNode, ActionNode, CallFlow
├── nlp/
│   ├── classifier.py  # Clasificador de intenciones (TF-IDF + nearest centroid)
│   └── preprocessing.py  # Preprocesamiento de texto español
├── tts/
│   └── responses.py   # Generador de TwiML (Text-to-Speech)
├── api/
│   └── server.py      # Servidor FastAPI con webhooks para Twilio
└── config.py          # Configuración global

data/
├── call_flow.json     # Definición del flujo de llamada
└── training_data.json # Datos de entrenamiento para el clasificador

scripts/
├── train.py           # Entrenar el modelo de intenciones
├── serve.py           # Iniciar el servidor API
└── simulate.py        # Simulador interactivo de llamadas
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

### 1. Entrenar la IA

```bash
python scripts/train.py
```

Salida esperada:
```
Training data: 97 examples
Intents:        6
Vocabulary:     143 words
Train accuracy: 100.0%
Model saved to: data/intent_model.json
```

### 2. Probar con el simulador

```bash
python scripts/simulate.py
```

Ejemplo de sesión:
```
📞 Sistema: Bienvenido a Danooc. Gracias por llamarnos.
   Presione 1 para ventas. Presione 2 para soporte técnico...

Tú: 1
📞 Sistema: Ha seleccionado ventas. Presione 1 para información de productos...

Tú: necesito una cotización
  [NLP] intent=ventas, confianza=85%
📞 Sistema: Para solicitar una cotización, deje su mensaje...
```

### 3. Iniciar el servidor (para Twilio)

```bash
python scripts/serve.py --port 8000
```

## Intenciones incluidas

| Intent | Ejemplos |
|---|---|
| `ventas` | "quiero comprar", "cotización", "precios" |
| `soporte` | "tengo un problema", "no funciona", "falla" |
| `facturacion` | "necesito mi factura", "cuánto debo", "pagar" |
| `cancelacion` | "quiero cancelar", "dar de baja" |
| `agente` | "hablar con alguien", "operador" |
| `horarios` | "a qué hora abren", "horarios de atención" |

## Flujo de llamada

```
Llamada entrante
  └─ Menú principal
       ├─ 1: Ventas
       │    ├─ 1: Info productos
       │    ├─ 2: Cotización (buzón de voz)
       │    └─ 0: Transferir a ventas
       ├─ 2: Soporte
       │    ├─ 1: Reportar falla (buzón de voz)
       │    ├─ 2: Estado de ticket → transferir
       │    └─ 0: Transferir a técnico
       ├─ 3: Facturación
       │    ├─ 1: Consultar saldo → transferir
       │    ├─ 2: Solicitar factura → transferir
       │    └─ 0: Transferir a facturación
       ├─ 4: Cancelaciones
       │    ├─ 1: Agente de retención
       │    ├─ 2: Continuar cancelación
       │    └─ 9: Volver al menú
       ├─ 5: Horarios → info y colgar
       └─ 0: Hablar con agente → transferir
```

## Personalización

### Agregar intenciones nuevas

Edita `data/training_data.json` y agrega ejemplos:

```json
{"text": "quiero cambiar mi plan", "intent": "cambio_plan"}
```

Luego re-entrena: `python scripts/train.py`

### Modificar el flujo de llamada

Edita `data/call_flow.json` para agregar menús, cambiar prompts o agregar nuevos destinos de transferencia.

### Integración con Twilio

1. Inicia el servidor: `python scripts/serve.py --base-url https://tu-dominio.com`
2. En Twilio, configura el webhook de llamadas entrantes a: `https://tu-dominio.com/voice/incoming`

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| POST | `/voice/incoming` | Llamada entrante (webhook de Twilio) |
| POST | `/voice/handle-input` | Recibe DTMF o voz tras un Gather |
| POST | `/voice/no-input` | Timeout sin respuesta |
| POST | `/voice/voicemail` | Recibe grabación de buzón de voz |
| GET | `/health` | Health check |

## Licencia

MIT
