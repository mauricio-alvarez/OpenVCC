import timm


model = timm.make_model("")


# Logic to freeze all heads in every layer except one, so we are gonna have 12!
